import os
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Subset

from .preprocess_util import get_transforms
from .training_util import train_one_epoch, validate
from .model import DogCatCNN


def main():
    load_dotenv()

    train_path = os.getenv("filtered_train_data_location")
    test_path = os.getenv("input_test_data_location")

    if not train_path or not test_path:
        raise ValueError("Missing dataset path(s) in .env file")

    full_train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=get_transforms(train=True),
    )

    full_val_dataset = datasets.ImageFolder(
        root=train_path,
        transform=get_transforms(train=False),
    )

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_subset, val_subset = random_split(
        range(len(full_train_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataset = Subset(full_train_dataset, train_subset.indices)
    val_dataset = Subset(full_val_dataset, val_subset.indices)

    test_dataset = datasets.ImageFolder(
        root=test_path,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print("Classes:", full_train_dataset.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DogCatCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 3
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_custom_cnn.pth")
            print("  Saved new best model.")
        print()

    print("\nFinal evaluation on test set...\n")
    model.load_state_dict(torch.load("best_custom_cnn.pth", map_location=device))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()