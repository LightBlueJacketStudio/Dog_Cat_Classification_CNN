# Defines functions for image preprocessing

import os
from dotenv import load_dotenv
from typing import Tuple

from PIL import Image
import torch
from torchvision import transforms

from torchvision.transforms import ToPILImage


def load_and_augment_image(
    image_size: Tuple[int, int] = (224, 224),
    train: bool = True,
) -> torch.Tensor:
    """
    Load an image from the path stored in the `output_train_data_location`
    environment variable, preprocess it for CNN input, and return a torch.Tensor.

    Args:
        image_size: Target image size as (height, width).
        train: If True, apply data augmentation. If False, only resize/normalize.

    Returns:
        A torch.Tensor of shape [C, H, W].

    Raises:
        EnvironmentError: If the env var is not set.
        FileNotFoundError: If the image path does not exist.
        ValueError: If the file cannot be opened as an image.
    """
    image_path = os.getenv("test_input")

    if not image_path:
        raise EnvironmentError(
            "Environment variable 'output_train_data_location' is not set."
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image file not found at path: {image_path}"
        )

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to open image at {image_path}: {exc}") from exc

    if train:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    image_tensor = transform(image)
    return image_tensor


def save_tensor_image(tensor: torch.Tensor, save_path: str):
    """
    Save a tensor image (C, H, W) to disk.
    Automatically unnormalizes if needed.
    """
    # Unnormalize (if you used ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    tensor = tensor.clone().detach()
    tensor = tensor * std + mean  # unnormalize
    tensor = torch.clamp(tensor, 0, 1)

    to_pil = ToPILImage()
    image = to_pil(tensor)
    image.save(save_path)

if __name__ == "__main__":
    load_dotenv()
    tensor = load_and_augment_image()
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    save_tensor_image(tensor, "augmented_image.jpg")