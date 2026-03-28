import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import SimpleCNN
from utils.dataset import get_dataloaders

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
trainloader, testloader = get_dataloaders()

# Model
model = SimpleCNN().to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}")

print("Training complete")

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Baseline CNN Accuracy: {accuracy:.2f}%")

# Save results
with open("results/cnn_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}%\n")