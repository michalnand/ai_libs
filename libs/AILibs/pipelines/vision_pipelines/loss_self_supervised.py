import torch



class SIGRegLoss(torch.nn.Module):
    """
    SIGReg for generic features (no temporal dimension)
    Input: (B, D)
    """
    def __init__(self, knots=17, num_proj=512):
        super().__init__()
        self.num_proj = num_proj    

        # integration grid
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)

        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # trapezoidal rule

        # Gaussian characteristic function
        phi = torch.exp(-t.square() / 2.0)

        # windowing (same as paper)
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * window)

    def forward(self, z):
        """
        z: (B, D)
        """
        B, D = z.shape

        device = z.device

        # 1. sample random projection directions
        A = torch.randn(D, self.num_proj, device=device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-8)

        # 2. project → (B, M)
        proj = z @ A

        # 3. expand for t grid → (B, M, K)
        x_t = proj.unsqueeze(-1) * self.t.to(device)

        # 4. empirical characteristic function
        cos_term = torch.cos(x_t).mean(dim=0)  # (M, K)
        sin_term = torch.sin(x_t).mean(dim=0)  # (M, K)

        # 5. compute Epps–Pulley error
        err = (cos_term - self.phi.to(device)).pow(2) + sin_term.pow(2)

        # 6. integrate over t
        stat = err @ self.weights.to(device)  # (M,)

        # 7. scale by sample size (important!)
        #stat = stat * B

        # 8. average over projections
        return stat.mean()

