from typing import Tuple
from torchvision import transforms
from dotenv import load_dotenv
import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    """"
    Train the model for one epoch.
    Args:   model: The neural network model to train.
            loader: DataLoader for the training dataset.
            criterion: Loss function to optimize.
            optimizer: Optimization algorithm to update model weights.
            device: The device (CPU or GPU) to perform computations on.
    Returns:    
            Average loss
            Accuracy for the epoch.    
    """""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_loss = total_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc