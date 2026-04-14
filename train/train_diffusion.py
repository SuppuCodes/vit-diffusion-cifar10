import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
os.makedirs("results", exist_ok=True)

from models.unet import SimpleUNet
from utils.dataset import get_dataloaders
from utils.diffusion_utils import linear_beta_schedule, get_noise
from utils.diffusion_utils import sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainloader, _ = get_dataloaders()

model = SimpleUNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()

timesteps = 50  # SMALL for CPU
betas = linear_beta_schedule(timesteps).to(device)

epochs = 10  # keep small

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, _ in trainloader:
        images = images.to(device)

        t = torch.randint(0, timesteps, (images.size(0),), device=device)

        noisy_images, noise = get_noise(images, t, betas)

        predicted_noise = model(noisy_images, t)

        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.3f}")

print("Diffusion training complete")
model.eval()
samples = sample(model, (16, 3, 32, 32), timesteps, betas, device)

vutils.save_image(samples, "results/diffusion_samples.png", normalize=True)

print("Samples saved!")

# Save model
torch.save(model.state_dict(), "results/diffusion_model.pth")
print("Diffusion model saved!")