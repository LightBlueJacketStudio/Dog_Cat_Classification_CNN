from torchvision import datasets
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import os
from .preprocess_util import *
from .training_util import *
from .model import *

def main ():
    # load the env variable
    load_dotenv()

    # build dataset
    train_dataset = datasets.ImageFolder(
        root=os.getenv("filtered_train_data_location"),
        transform=get_transforms(train=True),
    )

    val_dataset = datasets.ImageFolder(
        root=os.getenv("input_test_data_location"),
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )

    print(train_dataset.classes)   # ['cat', 'dog']

    # define device and model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = CatDogCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training loop
    # evaluation loop
    # yea we're done

if __name__ == "__main__":
    main()