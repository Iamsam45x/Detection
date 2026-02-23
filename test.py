import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model.cnn_model import ForgeryCNN
from preprocessing.dataset_loader import ForgeryDataset

# Load dataset
dataset = ForgeryDataset("dataset")

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

_, _, test_set = random_split(dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

# Load model
model = ForgeryCNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()


criterion = nn.BCELoss()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.unsqueeze(1)
        outputs = model(images)
        preds = (outputs > 0.5).float()

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Convert to numpy
all_preds = [int(p[0]) for p in all_preds]
all_labels = [int(l[0]) for l in all_labels]

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)
