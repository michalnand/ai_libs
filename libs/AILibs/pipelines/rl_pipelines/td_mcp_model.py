import torch

class TDMPC2Model(torch.nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 1. Encoder: s_t -> z_t
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_dim, latent_dim)
        )
        
        # 2. Dynamics: (z_t, a_t) -> z_{t+1}
        self.dynamics = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + action_dim, hidden_dim),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_dim, latent_dim)
        )
        
        # 3. Reward Predictor: (z_t, a_t) -> r_t
        self.reward_predictor = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + action_dim, hidden_dim),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_dim, 1)
        )   
        
        # 4. Q-Value Predictor (Single for simplicity, usually an ensemble of 2 or 5)
        self.q_predictor = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + action_dim, hidden_dim),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_dim, 1)
        )

        self._init_weights(self.encoder)
        self._init_weights(self.dynamics)
        self._init_weights(self.reward_predictor)
        self._init_weights(self.q_predictor)
        

    def encode(self, x):
        # Shape: [B, state_dim] -> [B, latent_dim]
        z = self.encoder(x)
        # Simplified SimNorm: L2 normalization to bound the latent space
        z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z

    def next_latent(self, z, a):
        # Shape: [B, latent_dim], [B, action_dim] -> [B, latent_dim]
        za = torch.cat([z, a], dim=-1)
        z_next = self.dynamics(za)
        return torch.nn.functional.normalize(z_next, p=2, dim=-1)


    def _init_weights(self, m):    

        torch.nn.init.orthogonal_(m[0].weight, gain=0.5)
        torch.nn.init.zeros_(m[0].bias)

        torch.nn.init.orthogonal_(m[2].weight, gain=0.01)
        torch.nn.init.zeros_(m[2].bias)
