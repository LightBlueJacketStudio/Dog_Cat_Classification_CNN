import os
from typing import Tuple

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage


def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    train: bool = True,
):
    if train:
        return transforms.Compose([
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
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def load_and_preprocess_image(
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    train: bool = False,
) -> torch.Tensor:
    """
    Load one image from disk and return a preprocessed tensor [C, H, W].
    """
    if not image_path:
        raise ValueError("image_path is empty")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to open image at {image_path}: {exc}") from exc

    transform = get_transforms(image_size=image_size, train=train)
    return transform(image)


def load_from_env(
    env_key: str = "test_input",
    image_size: Tuple[int, int] = (224, 224),
    train: bool = False,
) -> torch.Tensor:
    """
    Load an image from the path stored in the `test_input`
    environment variable, preprocess it for CNN input,
    and return a torch.Tensor.
    """
    image_path = os.getenv(env_key)
    if not image_path:
        raise EnvironmentError(f"Environment variable '{env_key}' is not set.")
    return load_and_preprocess_image(image_path, image_size=image_size, train=train)


def save_tensor_image(tensor: torch.Tensor, save_path: str):
    """
    Save a tensor image (C, H, W) to disk.
    Automatically unnormalizes ImageNet-normalized tensors.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    tensor = tensor.clone().detach().cpu()
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)

    image = ToPILImage()(tensor)
    image.save(save_path)