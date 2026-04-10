import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import CNN
from utils.dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainloader, testloader = get_dataloaders()

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

epochs = 10

for epoch in range(epochs):
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

print("Training complete")