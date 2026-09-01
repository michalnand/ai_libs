import torch

class MLPResidual(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super().__init__()

        self.lin0 = torch.nn.Linear(n_inputs, n_hidden)
        self.act0 = torch.nn.SiLU()
        self.lin1 = torch.nn.Linear(n_hidden, n_inputs)

        torch.nn.init.orthogonal_(self.lin0.weight, gain=0.5)
        torch.nn.init.orthogonal_(self.lin1.weight, gain=0.01)

        torch.nn.init.zeros_(self.lin0.bias)
        torch.nn.init.zeros_(self.lin1.bias)    

    def forward(self, x):
        y = self.lin0(x)
        y = self.act0(y)    
        y = self.lin1(y)

        return y + x


class MLPModel(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers, n_hidden):
        super().__init__()

        self.proj_in = torch.nn.Linear(n_inputs, n_hidden)
        
        # Unpack the list into positional arguments using *
        self.hidden = torch.nn.Sequential(
            *[MLPResidual(n_hidden, 2*n_hidden) for _ in range(n_layers)]
        )

        self.proj_out = torch.nn.Linear(n_hidden, n_outputs)

        torch.nn.init.orthogonal_(self.proj_in.weight, gain=0.5)
        torch.nn.init.orthogonal_(self.proj_out.weight, gain=0.01)

        torch.nn.init.zeros_(self.proj_in.bias)
        torch.nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        z = self.proj_in(x)
        z = self.hidden(z)
        y = self.proj_out(z)

        return y