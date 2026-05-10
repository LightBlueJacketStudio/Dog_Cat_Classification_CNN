import os
from dotenv import load_dotenv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# Load env
load_dotenv()

train_path = os.getenv("FILTERED_TRAIN_DATA_LOCATION")

if not train_path:
    raise ValueError("Path not found in .env")

# Basic transform (no normalization yet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(
    root=train_path,
    transform=transform
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Compute mean and std
mean = 0.0
std = 0.0
total_images = 0

for images, _ in loader:
    batch_size = images.size(0)
    images = images.view(batch_size, images.size(1), -1)  # (B, C, H*W)

    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total_images += batch_size

mean /= total_images
std /= total_images

print(f"Mean: {mean}")
print(f"Std: {std}")