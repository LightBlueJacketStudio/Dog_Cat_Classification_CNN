import os
import random
from dotenv import load_dotenv
import shutil

#Filter out images smaller than 16KB to ensure quality of the dataset
#Filtered data will be saved in "dataset_filtered"

load_dotenv()

#input data
input_train_dir = os.getenv("input_train_data_location")
input_test_dir = os.getenv("input_test_data_location")

#output data
output_train_dir = os.getenv("filtered_train_data_location")
# output_test_dir = os.getenv("filtered_test_data_location")
TRAIN_SAMPLE = 5000 # 5000 data point each label dog/cat for training data
MIN_SIZE = 16 * 1024  # size in byte 16KB

for label in ["cats", "dogs"]:
    input_path = os.path.join(input_train_dir, label)
    output_path = os.path.join(output_train_dir, label)

    os.makedirs(output_path, exist_ok=True)

    qualified = [f for f in os.listdir(input_path)
                 if os.path.isfile(os.path.join(input_path, f))
                 and os.path.getsize(os.path.join(input_path, f)) > MIN_SIZE]
    sampled = random.sample(qualified, min(TRAIN_SAMPLE, len(qualified)))

    for file in sampled:
        shutil.copy(os.path.join(input_path, file), os.path.join(output_path, file))
    print(f'successfully filtered and sampled {len(sampled)} {label}')
