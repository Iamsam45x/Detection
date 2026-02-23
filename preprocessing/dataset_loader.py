import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

IMAGE_SIZE = 224

class ForgeryDataset(Dataset):
    def __init__(self, base_path):
        self.data = []
        self.labels = []

        for label, folder in enumerate(['original', 'tampered']):
            folder_path = os.path.join(base_path, folder)
            for file in os.listdir(folder_path):
                self.data.append(os.path.join(folder_path, file))
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(self.data[idx])
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW

        return torch.tensor(image, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
