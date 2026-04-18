import os
from dotenv import load_dotenv
import shutil

#Filter out images smaller than 16KB to ensure quality of the dataset
#Filtered data will be saved in "dataset_filtered"

load_dotenv()

#input data
input_train_dir = os.getenv("input_train_data_location")
input_test_dir = os.getenv("input_test_data_location")

#output data
output_train_dir = os.getenv("output_train_data_location")
output_test_dir = os.getenv("output_test_data_location")

MIN_SIZE = 16 * 1024  # size in byte 16KB

for label in ["cats", "dogs"]:
    input_path = os.path.join(input_train_dir, label)
    output_path = os.path.join(output_train_dir, label)

    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(input_path):
        file_path = os.path.join(input_path, file)

        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)

            if size > MIN_SIZE:
                shutil.copy(file_path, os.path.join(output_path, file))
    print('successfully filtered')
