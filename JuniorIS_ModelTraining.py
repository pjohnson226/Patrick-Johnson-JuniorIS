from google.colab import drive
drive.mount("/content/drive")

#The above code does the following:
#Imports the drive module from the google.colab package which allows me to mount my Google Drive to the Colab environment to access my files and to save results, models etc...

!pip install datasets
!pip install accelerate
!pip install transformers
import accelerate
import transformers

#The above code does the following:
#!pip is used to interact with the python package manager to install packages directly in my notebook.
#The datasets library provides access to NLP datasets and tasks. The accelerate library provides tools for accelerating deep learing training (a requirement of using PyTorch with HuggingFace's Trainer as seen later). The Transformer library from HuggingFace provides access to transformer-based models and tools for working with pre-trained language models. The rest of the lines are imports needed to make these libraries functional in the notebook.


# Datasets load_dataset function for Huggingface
from datasets import load_dataset
# Pretty print (slightly more intuitive than regular print output)
from pprint import pprint
# Necessary Transformer Imports
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments
# PyTorch Imports
import torch
from torch.utils.data import DataLoader
# For random sampling
import random

#The above code does the following:
#Imports load_dataset from datasets which allows for loading datasets from HuggingFace as seen in the next code block. Imports pretty print which allows for printing data structures in a nicer format than the standard print(). Imports necessary packages for working with and training Transformers. Imports torch which is the core PyTorch library for tensor computation. Imports DataLoader which is used to load data in batches for training in PyTorch. Imports random which is used for random sampling of filtered data as seen later.

dataset_dict = load_dataset('HUPD/hupd',
    name='sample',
    data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather",
    icpr_label=None,
    train_filing_start_date='2016-01-01',
    train_filing_end_date='2016-01-21',
    val_filing_start_date='2016-01-22',
    val_filing_end_date='2016-01-31',
    #cache_dir="/content/drive/My Drive/HUPD"
)

#The above code does the following:
#Loads the sample version of the HUPD dataset from HuggingFace. This is a much smaller version of the HUPD dataset, and its purpose is primarily for debugging. 
#The dates signify the start and end dates of the patents desired which are then assigned to their respective train and validation variables. 
#Cacheing the dataset to google drive is currently commented out.

print(dataset_dict) # view downloaded HUPD data dictionary

print(f'Train dataset size: {dataset_dict["train"].shape}')
print(f'Validation dataset size: {dataset_dict["validation"].shape}')

#The above code does the following:
#Accesses the train and val keys of the dataset dictionary and gets the shape (dimensions) of the datastet. Train has 16,153 samples and 14 features.

# Label-to-index mapping for the decision status field
decision_to_str = {'REJECTED': 0, 'ACCEPTED': 1, 'PENDING': 2, 'CONT-REJECTED': 3, 'CONT-ACCEPTED': 4, 'CONT-PENDING': 5}

# Helper function
def map_decision_to_string(example):
    return {'decision': decision_to_str[example['decision']]}

#The above code does the following:
#The decision_to_str variable maps patent decision statuses to an integer for downstream machine interpretation.
#Map_decision_to_string then iterates through each sample in the dattset and modies the string label in the decision feature to now be the integer it was assigned in decision_to_str.

# Re-labeling/mapping decision labels
train_set = dataset_dict['train'].map(map_decision_to_string)
val_set = dataset_dict['validation'].map(map_decision_to_string)

#^This code calls the helper function to relabel for the train and val dataset dictionaries.

# Prompt the user to input the name of the model
model_name = input("Enter the name of the model for the tokenizer (options: 'bert-base-uncased', 'distilbert-base-uncased', and 'roberta-base': ")

#For accepted and rejected (after filtering out other lables below)
Classes = 2
#Model and Tokenizer Creation
def create_model_and_tokenizer(model_name = model_name, section='background', n_classes=Classes, max_length=512, train_set=train_set, val_set=val_set):
  if model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base']:
    config = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes, output_hidden_states=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.max_length = max_length
    tokenizer.model_max_length = max_length
    train_set = train_set.map(
      lambda e: tokenizer((e[section]), truncation=True, padding='max_length'),
      batched=True)
    val_set = val_set.map(
      lambda e: tokenizer((e[section]), truncation=True, padding='max_length'),
      batched=True)
    model = config
  else:
    raise ValueError(f"Model {model_name} not supported")
  return model, tokenizer, train_set, val_set, config

model, tokenizer, train_set, val_set, config = create_model_and_tokenizer(model_name)

#The above code does the following:
#Accepts as input the transformer-based model to be created by the function create_model_and_tokenizer. Defines the number of classes the model will try to predict (2). 
#The function takes various parameters that are necessary for model creation using the HuggingFace packages AutoModelForSequenceClassifcation, AutoConfig, and AutoTokenizer. 
#The section specifies which section of the patent should be included in the train_set and the val_set and is the section the model will learn on for the classification task. 
#The max_length specifies the max length of the input sequence. So, if the background is over 512 tokens, the rest will be truncated. 
#If the input sequences are shorter then max_length they are padded (with nonsense fluff) to match the max_length for input uniformity. 
#The train_set and val_set are then shortened and tokenized to only include the specified section. 
#Importantly, the function uses the Auto-based tools from HuggingFace which will automatically choose the proper model, configuration, and tokenizer based on the inputted model_name. 
#Various objects are then returned for subsequent data filtering and model training.

print(model)
print(tokenizer)
print(train_set)
print(val_set)

# Set the format (converting to PyTorch Tensors, defining column list for classification task)
train_set.set_format(type='torch',
    columns=['input_ids', 'attention_mask', 'decision'])

val_set.set_format(type='torch',
    columns=['input_ids', 'attention_mask', 'decision'])

#^This code sets the format of the train and validation sets to be compatible with PyTorch tensors and specifies which columns should be included in the processed dataset.

# A helper function that converts ids into tokens
def convert_ids_to_string(tokenizer, input):
    return ' '.join(tokenizer.convert_ids_to_tokens(input))

#^This code converts the tokenized inputs back into human readable format.

# Filter out labels 2, 3, 4 and 5
filtered_train_set = train_set.filter(lambda example: example['decision'] not in [2, 3, 4, 5])
filtered_val_set = val_set.filter(lambda example: example['decision'] not in [2, 3, 4, 5])

# Number of accepted and rejected
num_accepted = len(filtered_train_set.filter(lambda example: example['decision'] == 0))
num_rejected = len(filtered_train_set.filter(lambda example: example['decision'] == 1))
print(f'Accepted: {num_accepted}')
print(f'Rejected: {num_rejected}')

#^This code filters out the unwanted decision labels and then determines the number of accepted (0) and Rejected (1) labels remaining.

# Separate data by label (0 and 1)
accepted_examples = [ex for ex in filtered_train_set if ex['decision'] == 0]
rejected_examples = [ex for ex in filtered_train_set if ex['decision'] == 1]

# Calculate the desired split size (50% of the smaller group)
split_size = min(len(accepted_examples), len(rejected_examples)) // 2

# Randomly sample split_size elements from each label group
train_accepted = random.sample(accepted_examples, split_size)
train_rejected = random.sample(rejected_examples, split_size)

# Combine train examples while maintaining order
balanced_train_set = train_accepted + train_rejected

# Print the sizes of the balanced train set
print(f"Balanced train set size: {len(balanced_train_set)}")

#^This code then seperates the accepted and rejected of filtered_train_set into their respective variables. 
#Determines a split size to allocate 50% accepted and 50% rejected into the balanced_train_set by concatenating train_accepted and train_rejected which contains equal amount of samples.

# Calculate the number of accepted and rejected claims in the balanced_train_set
num_accepted_balanced = len([ex for ex in balanced_train_set if ex['decision'] == 0])
num_rejected_balanced = len([ex for ex in balanced_train_set if ex['decision'] == 1])

# Confirmation of 50/50 split
print(f"Number of accepted claims in balanced train set: {num_accepted_balanced}")
print(f"Number of rejected claims in balanced train set: {num_rejected_balanced}")

#^This code confirms the desired balance of 50/50.

# Calculate the test set size (15% of the total data)
test_size_balanced = int(0.15 * len(balanced_train_set))

# Create the test set by taking the remaining examples
test_set_balanced = balanced_train_set[-test_size_balanced:]

# Remove the test examples from the balanced train set
train_set_balanced = balanced_train_set[:-test_size_balanced]

# Print the sizes of balanced train and test sets
print(f"Balanced train set size: {len(train_set_balanced)}")
print(f"Balanced test set size: {len(test_set_balanced)}")

#^The data is then further filtered to obtain an 85/15 train test split where 85% of the data goes into train_set_balanced and 15% into test_size_balanced.

# Confirmation of desired format after filtering
pprint(test_set_balanced[0])

if train_set_balanced:
    sample = train_set_balanced[0]  # Get the first sample
    column_names = list(sample.keys())  # Extract column names from keys of the sample dictionary
    print("Column names of train_set_balanced:", column_names)
else:
    print("Train set is empty.")

#^This code shows the column names of the train_set_balanced.

# Rename 'decision' column to 'labels' in train_set_balanced
for sample in train_set_balanced:
    if 'decision' in sample:
        sample['labels'] = sample.pop('decision')

# Rename 'decision' column to 'labels' in test_set_balanced
for sample in test_set_balanced:
    if 'decision' in sample:
        sample['labels'] = sample.pop('decision')

#^This code renames decision to labels to fit the desired format of the classification model (needs attention_mask, input_ids, and labels).

# Confirmation of succesful column rename
if train_set_balanced:
    sample = train_set_balanced[0]  # Get the first sample
    column_names = list(sample.keys())  # Extract column names from keys of the sample dictionary
    print("Column names of train_set_balanced:", column_names)
else:
    print("Train set is empty.")

# Confirmation that rename reflected in individual sample
pprint(train_set_balanced[0])

from sklearn.metrics import accuracy_score

#function to compute accuracy
def compute_accuracy(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {'accuracy': accuracy_score(labels, preds)}

#^This code defines a function using scikit learn library to calculate accuracy to track model performance.

!pip install optuna

#^Optuna is a library that allows for hyperparameter optimization with HuggingFace trainer. 
#These trials iterate through a given range of hyperparameters to determine the optimal set of hyerparameters.

import optuna
from optuna import Trial

def define_search_space(trial):
  return{
    "learning_rate": trial.suggest_float("learning_rate", low=1e-5, high=5e-5), # pace of internal parameter updating
    "per_device_train_batch_size": trial.suggest_int("per_device_train_batch_size", low=8, high=16), # how much data is processed each training step
    "num_train_epochs": trial.suggest_int("num_train_epochs", low=3, high=5), # number of times model trains on dataset
    "weight_decay": trial.suggest_float("weight_decay", low=.001, high=.1), # strength of model penalty, used to prevent overfitting
    "per_device_eval_batch_size": trial.suggest_int("per_device_eval_batch_size", low=8, high=32) # how much data is processed each evaluation step
  }

#^This code defines the search space which specifies the hyperparameters to trial and the ranges that should be tested.

def model_init(trial):
  return model


#^This code defines the model_init function which, normally, would instantiate the model for trial runs using Optuna. 
#However, the model has already been instantated earlier in the create_model_and_tokenizer function so it simply returns model.

# training arguments
training_args = TrainingArguments(
    evaluation_strategy='epoch',  # Evaluate at the end of each epoch
    output_dir = "/content/gdrive/MyDrive/HUPD",
    save_strategy='epoch',
    load_best_model_at_end=True,

)

# Instantiating Trainer
trainer = Trainer(
    args=training_args,                  # Training arguments, defined above
    train_dataset=train_set_balanced,    # Training dataset
    eval_dataset=test_set_balanced,      # Evaluation dataset
    tokenizer=tokenizer,
    compute_metrics=compute_accuracy,
    model_init=model_init
)

best_run = trainer.hyperparameter_search(
    n_trials=3,
    direction="maximize", #maximize accuracy
    hp_space=define_search_space,
    backend="optuna",
    sampler=optuna.samplers.TPESampler(),
)

# Train model
model = trainer.model

#^This code defines the training and evaluation processes. 
#The Trainer and TrainingArguments tools from HuggingFace are used to train and evaluate the model using Optuna hyperparameter optimization. 
#Best run contains the best hyperparameter configuration.
