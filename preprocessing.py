import os
import shutil

#Filter out images smaller than 16KB to ensure quality of the dataset
#Filtered data will be saved in "dataset_filtered"

input_dir = "dataset/test"
output_dir = "dataset_filtered/test"

MIN_SIZE = 16 * 1024  # size in byte 16KB

for label in ["cats", "dogs"]:
    input_path = os.path.join(input_dir, label)
    output_path = os.path.join(output_dir, label)

    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(input_path):
        file_path = os.path.join(input_path, file)

        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)

            if size > MIN_SIZE:
                shutil.copy(file_path, os.path.join(output_path, file))
