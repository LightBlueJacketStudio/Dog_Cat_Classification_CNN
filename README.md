# Dog and Cat Classification CNN
A repo for building a CNN from scratch to classify Cats and Dogs in a picture with varying qualities

# Dataset
Dogs vs. Cats Image Classification Dataset<br>
<a href='https://www.kaggle.com/datasets/princelv84/dogsvscats/data'>Link</a>

# Project Proposal
Project ideas and methodology <br>
<a href='https://docs.google.com/document/d/1NS67avLl44VclIQg-QJNikpXlAA0dSK8Lv49odQ7LwU/edit?usp=sharing'>Link</a>

# Project Backlog
Tasks monitoring and milestones <br>
<a href='https://docs.google.com/spreadsheets/d/1H8CPgRGdHlObjSIvUn7mRfApElcvLQKXKnGzgNe0Kps/edit?usp=sharing'>Link</a>


# Set up

Start a virtual environment (this requires you to have a python 3.11 installed):

```
py -3.11 -m venv .venv
```

Activate the virtual environment (from project root):

```
.\.venv\Scripts\Activate.ps1
```

Install the required libraries:

```
pip install -r requirements.txt
```

Set up your environment variable

<li> create a file named `.env` in project root

<li> inside the `.env`, set up the correct variables

You should have your training and testing data folder

in the structure of 

```
<folder name>
| - train
| - |- dog
| - |- cat
|
| - test
| - |- dog
| - |- cat
```

your `.env` should look like
```
input_train_data_location="<path to the whole train dataset>"
input_test_data_location="<path to the whole test dataset>"

output_train_data_location="<path to the filtered train dataset>"
output_test_data_location="<path to the filtered test dataset>"
```

# Filter the original dataset to a smaller subset

We used file size as the filter (keeping size > 16 kB)

run 
```
python .\preprocessing.py
```
you should see a filtered dataset folder being created, this would be the data for the model to train


# Train the Pre-Trained model

Train the model using the training dataset, 

the script will automatically split the training set into a 80%, 20% segments 

and use the 20% segment as validation set.

Running the training script might take longer time if running on GPU,

I waited about 1 hour for my laptop to train on the dataset

```
python .\infr_EfficientNet.py
```
Once done, you should see a `model.pth` in the project root, 

that is the trained weight


# Use the Trained model

Test your trained efficientnet instance by running
```
python .\use_trained_eff_net.py
```

the end result should be `Dog (1.0000)`, where 1.0000 represents the probabilty