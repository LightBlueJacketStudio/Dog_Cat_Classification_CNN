import timm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

print("Script started")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 1)

model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

cat_image_path = "test_input/cat.554.jpg"
dog_image_path = "test_input/dog.67.jpg"

cat_image = Image.open(cat_image_path).convert("RGB")
cat_image = transform(cat_image).unsqueeze(0).to(device)

dog_image = Image.open(dog_image_path).convert("RGB")
dog_image = transform(dog_image).unsqueeze(0).to(device)

print("Image loaded")
print("Running inference")

with torch.no_grad():
    output = model(dog_image)
    prob = torch.sigmoid(output).item()

if prob > 0.5:
    print(f"Dog ({prob:.4f})")

else:
    print(f"Cat ({1 - prob:.4f})")


