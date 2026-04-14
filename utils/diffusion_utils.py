import torch

def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

def get_noise(x, t, betas):
    noise = torch.randn_like(x)

    alpha = 1.0 - betas
    alpha_hat = torch.cumprod(alpha, dim=0)

    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]

    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

def sample(model, shape, timesteps, betas, device):
    model.eval()
    x = torch.randn(shape).to(device)

    with torch.no_grad():
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

            predicted_noise = model(x, t_tensor)

            beta = betas[t]
            x = (x - beta * predicted_noise) / torch.sqrt(1 - beta)

            if t > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)

    return x