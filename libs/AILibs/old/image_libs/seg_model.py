import torch


class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, h_ch, stride = 1):
        super(Residual, self).__init__()

        self.conv0 = torch.nn.Conv2d(in_ch, h_ch, kernel_size=3, stride=stride, padding=1)
        self.act0  = torch.nn.SiLU()
        self.conv1 = torch.nn.Conv2d(h_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.act1  = torch.nn.SiLU()

        torch.nn.init.orthogonal_(self.conv0.weight, 0.5)
        torch.nn.init.zeros_(self.conv0.bias)
        torch.nn.init.orthogonal_(self.conv1.weight, 0.5)
        torch.nn.init.zeros_(self.conv1.bias)

        if in_ch != out_ch or stride != 1:
            self.conv_bp = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0)

            torch.nn.init.orthogonal_(self.conv_bp.weight, 0.5)
            torch.nn.init.zeros_(self.conv_bp.bias)
        else:
            self.conv_bp = None

    def forward(self, x):
        y = self.conv0(x)
        y = self.act0(y)
        y = self.conv1(y)

        if self.conv_bp is not None:
            x_ = self.conv_bp(x)
        else:
            x_ = x

        y = self.act1(y + x_)

        return y


class Outputhead(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):  
        super(Outputhead, self).__init__()  

        self.conv0 = torch.nn.Conv2d(n_inputs, n_hidden, kernel_size=1, stride=1, padding=0)
        self.act0  = torch.nn.SiLU()
       
        self.conv1 = torch.nn.Conv2d(n_hidden, n_outputs, kernel_size=1, stride=1, padding=0)

        torch.nn.init.orthogonal_(self.conv0.weight, 0.5)
        torch.nn.init.zeros_(self.conv0.bias)
        torch.nn.init.orthogonal_(self.conv1.weight, 0.01)
        torch.nn.init.zeros_(self.conv1.bias)

    def forward(self, x):
        y = self.conv0(x)
        y = self.act0(y)
        
        y = self.conv1(y)
      
        return y



class SegModel(torch.nn.Module):
    def __init__(self, input_shape, n_outputs):
        super(SegModel, self).__init__()

        self.l0 = torch.nn.Sequential(
            Residual(input_shape[0], 16, 16, 2),
            Residual(16, 16, 16)
        )

        self.l1 = torch.nn.Sequential(
            Residual(16, 32, 32, 2),
            Residual(32, 32, 32),
        )       

        self.l2 = torch.nn.Sequential(
            Residual(32, 64, 64, 2),
            Residual(64, 64, 64)
        )

        self.seg_head = Outputhead(16 + 32 + 64, 128, n_outputs)
   
   
    def forward(self, x):
        z_l0   = self.l0(x)
        z_l1   = self.l1(z_l0)
        z_l2   = self.l2(z_l1)

        z_l1 = torch.nn.functional.interpolate(z_l1, scale_factor=2, mode='nearest')
        z_l2 = torch.nn.functional.interpolate(z_l2, scale_factor=4, mode='nearest')

        z = torch.concatenate([z_l0, z_l1, z_l2], dim=1)
        
        seg = self.seg_head(z)

        return seg



