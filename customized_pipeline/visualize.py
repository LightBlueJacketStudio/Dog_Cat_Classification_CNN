# visualize_cnn.py

import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from dotenv import load_dotenv

from .model import DogCatCNN
from .preprocess_util import get_transforms


def save_feature_maps(feature_maps, layer_name, output_dir, max_maps=16):
    os.makedirs(output_dir, exist_ok=True)

    feature_maps = feature_maps.detach().cpu()

    # Shape: [batch, channels, height, width]
    feature_maps = feature_maps[0]

    num_maps = min(max_maps, feature_maps.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(f"Feature Maps: {layer_name}")

    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            fmap = feature_maps[i]
            ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"Map {i}")
        ax.axis("off")

    save_path = os.path.join(output_dir, f"{layer_name}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved {save_path}")


def save_input_image(image_tensor, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image = image_tensor.detach().cpu()[0]

    # Convert CHW to HWC
    image = image.permute(1, 2, 0)

    # Undo approximate normalization only if your transform normalized to [-1, 1]
    image = image.clamp(0, 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    save_path = os.path.join(output_dir, "00_input_image.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved {save_path}")


def visualize_model_layers(model, image_tensor, output_dir="cnn_visualizations"):
    model.eval()

    save_input_image(image_tensor, output_dir)

    x = image_tensor

    layer_counter = 1

    with torch.no_grad():
        for layer in model.features:
            x = layer(x)

            if isinstance(layer, torch.nn.Conv2d):
                layer_name = f"{layer_counter:02d}_conv_{layer.out_channels}_filters"
                save_feature_maps(
                    feature_maps=x,
                    layer_name=layer_name,
                    output_dir=output_dir,
                )
                layer_counter += 1

        pooled = model.classifier[0](x)
        flattened = model.classifier[1](pooled)
        hidden = model.classifier[2](flattened)
        activated = model.classifier[3](hidden)
        dropped = model.classifier[4](activated)
        output = model.classifier[5](dropped)

        probability = torch.sigmoid(output).item()
        prediction = "dog" if probability >= 0.5 else "cat"

        print("\nFinal Output")
        print("-" * 40)
        print(f"Raw logit   : {output.item():.4f}")
        print(f"Probability : {probability:.4f}")
        print(f"Prediction  : {prediction}")


from PIL import Image


def main():
    load_dotenv()

    image_path = os.getenv("test_input_cat")
    model_path = "best_custom_cnn.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image
    image = Image.open(image_path).convert("RGB")

    transform = get_transforms(train=False)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Optional: infer label from filename (just for display)
    if "cat" in image_path.lower():
        true_label = "cat"
    elif "dog" in image_path.lower():
        true_label = "dog"
    else:
        true_label = "unknown"

    model = DogCatCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("True label:", true_label)
    print("Using device:", device)

    visualize_model_layers(
        model=model,
        image_tensor=image_tensor,
        output_dir="cnn_visualizations",
    )
    load_dotenv()

    test_path = os.getenv("test_input_cat")
    model_path = "withAug_best_custom_cnn.pth"

    if not test_path:
        raise ValueError("Missing input_test_data_location in .env file")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.ImageFolder(
        root=test_path,
        transform=get_transforms(train=False),
    )

    image_tensor, label = dataset[0]
    image_tensor = image_tensor.unsqueeze(0).to(device)

    model = DogCatCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("True label:", dataset.classes[label])
    print("Using device:", device)

    visualize_model_layers(
        model=model,
        image_tensor=image_tensor,
        output_dir="cnn_visualizations",
    )


if __name__ == "__main__":
    main()