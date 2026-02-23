import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model.cnn_model import ForgeryCNN
from preprocessing.dataset_loader import ForgeryDataset

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001

# Load dataset
dataset = ForgeryDataset("dataset")

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Model, loss, optimizer
model = ForgeryCNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        labels = labels.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

print("Training completed.")
# Save trained model
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
