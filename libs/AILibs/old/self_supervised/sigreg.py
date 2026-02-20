import torch


def sigreg_loss(z, t_range = 5.0, num_t_points=17, num_projections=32):
    N, K = z.shape

    t = torch.linspace(-t_range, t_range, num_t_points, device=z.device) 
    phi_target = torch.exp(-0.5 * t**2)
    
    # 1, Sample random directions
    A = torch.randn(K, num_projections, device=z.device)
    A = A / A.norm(dim=0, keepdim=True)  # normalize columns

    # 2, Project embeddings
    # (N, M)
    projections = z @ A

    # 3, Compute empirical characteristic function
    # shape: (N, M, T)
    x_t = projections.unsqueeze(-1) * t

    # e^{i t x}
    ecf = torch.exp(1j * x_t).mean(dim=0)  # (M, T)

    # 4, Weighted L2 distance
    diff = ecf - phi_target  # broadcast over M
    error = (diff.abs() ** 2) * phi_target  # Gaussian window

    # integrate over t
    integral = torch.trapz(error, t, dim=1)

    # scale by N (as in paper)
    loss = N * integral.mean()

    return loss.real




def evaluate_distribution(z, label):
    #loss_marginal = sigreg_loss(z, num_projections=16)  
    loss_all  = 0 #sigreg_loss_all(z)
    loss_proj = sigreg_loss(z, num_projections=16)
    #print(f"{label:30s}  All: {loss_all.item():8.4f}  Proj: {loss_proj.item():8.4f}")
    print(f"{label:30s}   Proj: {loss_proj.item():8.6f}")
    


if __name__ == "__main__":    
    batch_size = 4096
    n_features = 512

    print("=== High-D: Marginal vs Projection ===\n\n\n")
    z_gaussian = torch.randn((batch_size, n_features))
    evaluate_distribution(z_gaussian, "Pure Gaussian")

    z_uniform = torch.rand((batch_size, n_features))
    evaluate_distribution(z_uniform, "Uniform")

    for alpha in torch.linspace(0, 1, steps=6):
        gaussian = torch.randn(batch_size, n_features)
        laplace = torch.distributions.Laplace(0, 1).sample((batch_size, n_features))
        z = (1 - alpha) * gaussian + alpha * laplace
        evaluate_distribution(z, f"Heavy tail alpha={alpha:.2f}")

    for alpha in torch.linspace(0, 1, steps=6):
        gaussian = torch.randn(batch_size, n_features)
        skewed = gaussian + alpha * (gaussian ** 2)
        evaluate_distribution(skewed, f"Skewed alpha={alpha:.2f}")

    for scale in [100.0, 10.0, 2.0, 1.0, 0.5, 0.1, 0.01, 0.0]:
        z = torch.randn(batch_size, n_features) * scale
        evaluate_distribution(z, f"Collapsed scale={scale}")

    for ratio in [0.01, 0.1, 1, 2, 5, 10, 20]:
        z = torch.randn(batch_size, n_features)
        z[:, 0] *= ratio
        evaluate_distribution(z, f"Anisotropy ratio={ratio}")