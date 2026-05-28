import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2)
        self.act = torch.nn.LeakyReLU(0.01)     

        torch.nn.init.orthogonal_(self.conv.weight, gain=0.5)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.act(self.conv(x))


class UnetModel(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_ch, 8, 3)
        self.enc2 = ConvBlock(8, 16, 3)
        self.enc3 = ConvBlock(16, 16, 3)
        self.enc4 = ConvBlock(16, 16, 5)
        self.enc5 = ConvBlock(16, 16, 7)
        self.enc6 = ConvBlock(16, 16, 7)

        self.pool = torch.nn.MaxPool2d(2)

        # Decoder
        self.up5 = torch.nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.dec5 = ConvBlock(32, 16, 7)

        self.up4 = torch.nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.dec4 = ConvBlock(32, 16, 5)

        self.up3 = torch.nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.dec3 = ConvBlock(32, 16, 3)

        self.up2 = torch.nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.dec2 = ConvBlock(32, 16, 3)

        self.up1 = torch.nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec1 = ConvBlock(16, 8, 3)

        # Final output
        self.out = torch.nn.Conv2d(8, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))

        # Decoder
        d5 = self.up5(e6)
        d5 = self.dec5(torch.cat([d5, e5], dim=1))

        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)
    

if __name__ == "__main__":
    model = UnetModel(3, 1)

    print(model)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)  
    print(out.shape)