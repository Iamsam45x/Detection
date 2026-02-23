import os
import cv2
import numpy as np

IMAGE_SIZE = 224

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    return image

def load_dataset(base_path):
    images = []
    labels = []

    for label, folder in enumerate(['original', 'tampered']):
        folder_path = os.path.join(base_path, folder)

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            image = preprocess_image(img_path)
            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    X, y = load_dataset(DATASET_DIR)
    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)
