import torch
import torch.nn as nn
import torch.optim as optim
import os
os.makedirs("results", exist_ok=True)

from models.vit import ViT
from utils.dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainloader, testloader = get_dataloaders()

model = ViT().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

epochs = 5  # keep it reasonable for CPU

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.3f}")
    scheduler.step()

print("Training complete")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"ViT Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "results/vit_model.pth")
print("ViT model saved!")

with open("results/vit_results.txt", "w") as f:
    f.write(f"ViT Accuracy: {100 * correct / total:.2f}%\n")