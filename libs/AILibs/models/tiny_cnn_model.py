import torch
import math

class CNNBlock(torch.nn.Module):

    def __init__(self, in_ch, h_ch, out_ch, stride = 1):
        super().__init__()

        self.norm_0 = torch.nn.InstanceNorm2d(in_ch, affine=True)

        self.conv0 = torch.nn.Conv2d(in_ch, h_ch, kernel_size=3, stride=stride, padding=1)
        self.act0  = torch.nn.SiLU()

        self.conv1 = torch.nn.Conv2d(h_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.act1  = torch.nn.SiLU()

        # Gain sqrt(2) compensates for the signal loss through the upcoming SiLU
        torch.nn.init.orthogonal_(self.conv0.weight, gain=math.sqrt(2))
        torch.nn.init.zeros_(self.conv0.bias)   

        # Gain 1.0 (or even smaller, like 0.1) is best for the end of a residual branch
        # so it doesn't blow up the variance when added to the shortcut
        torch.nn.init.orthogonal_(self.conv1.weight, gain=1.0)
        torch.nn.init.zeros_(self.conv1.bias)   

        if in_ch != out_ch or stride != 1:
            self.conv_bp = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0)
            # Shortcut connections should preserve the signal exactly
            torch.nn.init.orthogonal_(self.conv_bp.weight, gain=1.0)
            torch.nn.init.zeros_(self.conv_bp.bias)
        else:
            self.conv_bp = None
            

    def forward(self, x):
        x = self.norm_0(x)

        y = self.conv0(x)
        y = self.act0(y)

        y = self.conv1(y)

        if self.conv_bp is not None:
            y = y + self.conv_bp(x)
        else:
            y = y + x

        y = self.act1(y)

        return y

class MLPModel(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super().__init__()

        self.lin_0 = torch.nn.Linear(num_inputs, num_hidden)
        self.act_0 = torch.nn.SiLU()
        self.lin_1 = torch.nn.Linear(num_hidden, num_inputs)

        torch.nn.init.orthogonal_(self.lin_0.weight, gain=0.5)
        torch.nn.init.zeros_(self.lin_0.bias)

        torch.nn.init.orthogonal_(self.lin_1.weight, gain=0.01)
        torch.nn.init.zeros_(self.lin_1.bias)

    def forward(self, x):
        y = self.lin_0(x)
        y = self.act_0(y)
        y = self.lin_1(y)

        return y


class TinyCNNModel(torch.nn.Module):
    def __init__(self, in_ch = 3, num_features = 128):
        super().__init__()

        self.conv_in = torch.nn.Conv2d(in_ch, 32, kernel_size=7, stride=2, padding=7//2)
        self.act     = torch.nn.SiLU()

        self.b0 = CNNBlock(32, 64,  64, 2)  
        self.b1 = CNNBlock(64, 128, 128, 2)
        self.b2 = CNNBlock(128, 256, 128, 1)
        self.b3 = CNNBlock(128, 2*num_features, 2*num_features, 1)  

        self.conv_out = torch.nn.Conv2d(2*num_features, num_features, kernel_size=1, stride=1, padding=0)

        # Gain sqrt(2) because it feeds directly into SiLU
        torch.nn.init.orthogonal_(self.conv_in.weight, gain=math.sqrt(2))
        torch.nn.init.zeros_(self.conv_in.bias)

        # Gain 1.0 because this is a final projection layer mapping to the metric space
        torch.nn.init.orthogonal_(self.conv_out.weight, gain=1.0)
        torch.nn.init.zeros_(self.conv_out.bias)    

        self.projector = MLPModel(num_features, 2*num_features)



    def forward(self, x):

        y = self.conv_in(x)
        y = self.act(y)

        y = self.b0(y)
        y = self.b1(y)
        y = self.b2(y)
        y = self.b3(y)

        y = self.conv_out(y)

        return y

    def forward_projector(self, z):
        return self.projector(z) + z