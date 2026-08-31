from .replay_buffer import *
from .td_mcp_model  import *

import torch
import copy

class Config:
    def __init__(self):
        self.gamma          = 0.99

        self.state_shape    = (11,)       # Example: Hopper-v4
        self.action_shape   = (3,)
        self.latent_dim     = 128
        self.hidden_dim     = 256

        self.buffer_size    = 100000
        self.batch_size     = 256
        self.horizon        = 5      # Planning horizon (H)
        self.mppi_samples   = 512    # Number of candidate trajectories
        self.learning_rate  = 3e-4
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_freq     = 50     # Train every N steps

        self.cem_iterations = 6        # How many times to refine the distribution
        self.cem_elites = 50           # Top K samples to keep (roughly 10% of N)
        self.cem_alpha = 0.1           # Momentum for mean/std updates (smoothing)

        self.model = TDMPC2Model(
                    state_dim=self.state_shape[0], 
                    action_dim=self.action_shape[0], 
                    latent_dim=self.latent_dim,
                    hidden_dim=self.hidden_dim
                )



class TDMCPAgent:
    def __init__(self, env, config):
        self.env    = env
        self.config = config
        self.device = config.device
        
        self.buffer = ContinuousReplayBuffer(config.buffer_size, config.state_shape, config.action_shape)
        
        self.modle = config.model
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)


        self.model_target = copy.deepcopy(self.model)
        self.model_target.eval() # Never trains
        for param in self.model_target.parameters():
            param.requires_grad = False
            
        self.tau = 0.005 # Soft update rate
        
        self.n_steps    = 0
        self.gamma      = config.gamma

        # CEM State variables: We remember the plan from the previous step!
        # This gives us a massive head start for the next planning cycle.
        self.plan_mean = torch.zeros(config.horizon, config.action_shape[0], device=self.device)
        self.plan_std  = torch.ones(config.horizon, config.action_shape[0], device=self.device)

    @torch.no_grad()
    def plan(self, state):
        """ 
        Latent Planning using the Cross-Entropy Method (CEM).
        """
        self.model.eval()
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 1. Encode current state
        z_0 = self.model.encode(state_tensor) # [1, latent_dim]
        
        N = self.config.mppi_samples
        H = self.config.horizon
        num_elites = self.config.cem_elites
        
        # 2. Shift the previous plan forward by one step.
        # We executed action 0, so action 1 becomes the new action 0.
        # We pad the end with a zero mean and standard deviation of 1.0.
        self.plan_mean[:-1] = self.plan_mean[1:].clone()
        self.plan_mean[-1]  = 0.0
        
        self.plan_std[:-1] = self.plan_std[1:].clone()
        self.plan_std[-1]  = 1.0

        # Local copies for this specific planning loop
        mean = self.plan_mean.clone()
        std  = self.plan_std.clone()
        
        # 3. CEM Iteration Loop
        for iter_idx in range(self.config.cem_iterations):
            
            # Sample candidate action trajectories from the current Gaussian distribution
            # Shape: [N, H, action_dim]
            noise = torch.randn(N, H, self.config.action_shape[0], device=self.device)
            actions = mean + std * noise
            
            # Clamp actions to the valid environment range (assuming [-1, 1])
            actions = torch.clamp(actions, -1.0, 1.0)
            
            # Rollout variables
            z = z_0.repeat(N, 1) # [N, latent_dim]
            returns = torch.zeros(N, device=self.device)
            
            # Rollout in Latent Space
            for t in range(H):
                a_t = actions[:, t, :]
                r_t = self.model.reward_predictor(torch.cat([z, a_t], dim=-1)).squeeze(-1)
                returns += (self.gamma ** t) * r_t
                z = self.model.next_latent(z, a_t)
                
            # Add terminal Q-value
            q_final = self.model_target.q_predictor(torch.cat([z, actions[:, -1, :]], dim=-1)).squeeze(-1)
            returns += (self.gamma ** H) * q_final
            
            # 4. Find the Elites (Top K trajectories)
            # topk returns values and indices; we only need indices
            _, elite_idxs = torch.topk(returns, num_elites)
            elite_actions = actions[elite_idxs] # [num_elites, H, action_dim]
            
            # 5. Update the Distribution (Mean and Std) based on Elites
            new_mean = elite_actions.mean(dim=0)
            new_std  = elite_actions.std(dim=0)
            
            # Exponential Moving Average (Momentum) for stability
            # We don't instantly jump to the new distribution; we blend it.
            mean = (1.0 - self.config.cem_alpha) * mean + self.config.cem_alpha * new_mean
            
            # For the standard deviation, we usually decay it slightly faster or bound it
            # so the search narrows down over iterations.
            std  = (1.0 - self.config.cem_alpha) * std + self.config.cem_alpha * new_std
            std  = torch.clamp(std, min=0.01) # Prevent exact zero variance
            
        # 6. Save the final distribution parameters for the *next* environment step
        self.plan_mean = mean.clone()
        self.plan_std  = std.clone()
        
        # 7. Return the first action of the absolute best trajectory found
        # (Alternatively, you could return the first action of the final mean)
        best_idx = torch.argmax(returns)
        best_first_action = actions[best_idx, 0, :] 
        
        return best_first_action.cpu().numpy()

    
    def update(self):
        self.model.train()
        s_seq, a_seq, r_seq, d_seq = self.buffer.sample_sequence(
            self.config.batch_size, self.config.horizon, self.device
        )
        
        # Encode initial state s_0
        z_t = self.model.encode(s_seq[:, 0, :])

        # Self-Supervised Loss - optional
        if self._loss_ssl is not None:
            loss_ssl, loss_ssl_metric = self._loss_ssl(z_t)
        else:
            loss_ssl = 0.0

        
        loss_dyn = 0.0
        loss_rew = 0.0
        loss_q = 0.0
        
        for t in range(self.config.horizon):
            a_t = a_seq[:, t, :]
            
            # Predict
            r_pred = self.model.reward_predictor(torch.cat([z_t, a_t], dim=-1)).squeeze(-1)
            q_pred = self.model.q_predictor(torch.cat([z_t, a_t], dim=-1)).squeeze(-1)
            
            # True values (simplified Q-target)
            r_true = r_seq[:, t]
            # Use the TARGET network for the Q-target
            with torch.no_grad():
                z_next_true = self.model.encode(s_seq[:, t+1, :])
                q_next = self.model_target.q_predictor(torch.cat([z_next_true, a_seq[:, t+1, :]], dim=-1)).squeeze(-1)
                q_target = r_true + self.gamma * (1.0 - d_seq[:, t].float()) * q_next
                
            # Compute Latent Dynamics
            z_next_pred = self.model.next_latent(z_t, a_t)
            
            # Accumulate Losses
            loss_rew += torch.nn.functional.mse_loss(r_pred, r_true)
            loss_q   += torch.nn.functional.mse_loss(q_pred, q_target)
            loss_dyn += torch.nn.functional.mse_loss(z_next_pred, z_next_true.detach())
            
           
            # Step forward
            z_t = z_next_pred
            
        total_loss = loss_dyn + loss_rew + loss_q + (0.1 * loss_ssl)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()


        # soft update the target network
        with torch.no_grad():
            for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        # Return metrics dictionary
        metrics = {
            "loss/total": total_loss.item(),
            "loss/dynamics": loss_dyn.item() / self.config.horizon,
            "loss/reward": loss_rew.item() / self.config.horizon,
            "loss/q_value": loss_q.item() / self.config.horizon,
            "loss/loss_ssl": loss_ssl.item() / self.config.horizon, 
        }

        if self._loss_ssl is not None:
            metrics = metrics + loss_ssl_metric

        return  metrics

    def step(self, state):
        
        # Random exploration until buffer has enough data to plan
        if self.buffer.size < self.config.batch_size + self.config.horizon + 100:
            action = self.env.action_space.sample()
        else:
            action = self.plan(state)
            
        next_state, reward, done, _, _ = self.env.step(action)
        
        self.buffer.add(state, reward, action, done)
        self.n_steps += 1
        
        metrics = {}    
        if self.n_steps % self.config.train_freq == 0 and self.buffer.size > self.config.batch_size:
            metrics = self.update()
            
        return next_state, done, metrics