import torch


class MLPModel(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Shared or separate trunks. Separate is safer for PPO to avoid feature competition.
        self.actor_trunk = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )   
        
        # Two-headed Actor
        self.actor_mean = torch.nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = torch.nn.Linear(hidden_dim, action_dim)
        
        # Critic
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m): 
        if isinstance(m, torch.nn.Linear):
            # Orthogonal initialization is still best practice for RL
            torch.nn.init.orthogonal_(m.weight, gain=0.5)
            torch.nn.init.constant_(m.bias, 0.0)
            
        # Rescale the output layers to prevent massive initial gradients
        if m == self.actor_mean or m == self.actor_log_std: 
            torch.nn.init.orthogonal_(m.weight, gain=0.01)
        elif m == self.critic[-1]:
            torch.nn.init.orthogonal_(m.weight, gain=1.0) 

    def forward(self, x):
        actor_features = self.actor_trunk(x)
        
        mean = self.actor_mean(actor_features)
        log_std = self.actor_log_std(actor_features)
        
        # CRITICAL: Clamp state-dependent log_std to prevent numerical explosion 
        # or premature variance collapse in PPO.
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        value = self.critic(x).squeeze(-1)
        
        return mean, log_std, value