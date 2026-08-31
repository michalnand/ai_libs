import torch


class Config:
    def __init__(self):
        self.gamma          = 0.99

        self.state_shape    = (11,)       # Example: Hopper-v4
        self.action_shape   = (3,)

        self.entropy_beta       = 123
        self.eps_clip           = 123
        self.adv_coeff          = 123
        self.val_coeff          = 0.5
        
        
        self.num_steps      = 4096   #buffer size
        self.learning_rate  = 1e-4
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MLPModel(
                    state_dim=self.state_shape[0], 
                    action_dim=self.action_shape[0], 
                    hidden_dim=512,
                )

# envs already created in env = envpool.make(...)
class AgentPPO():
    def __init__(self, envs, Config):
        self.envs = envs


    def step(self, states):        

        return next_states