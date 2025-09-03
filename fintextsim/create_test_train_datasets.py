import pandas as pd
import numpy as np
import pickle
import os

os.chdir("/mnt/sdb1/home/simonj")

result_loc = "paper1/Data/train_test_sets/train_test_sets_2016_2023.pkl"

#------------Load labeled dataset
dataset_loc = "paper1/Data/Labeled_Dataset/labeled_dataset_2016_2023.pkl"

with open(dataset_loc, "rb") as file:
    ld = pickle.load(file)

ld_results = ld["dataset"]
labeled_dataset = ld_results["labeled_dataset"]
metadata = ld_results["metadata"]


from fintextsim.utils_fintextsim import create_test_dataset, prepare_triplet_data

train_set, test_set = create_test_dataset(labeled_dataset)

#create triplets
train_triplets = prepare_triplet_data(train_set)
test_triplets = prepare_triplet_data(test_set)

#---------------Transform  test and train-data into the required format: dataset with sentence and label 
from datasets import Dataset


train_dataset_triplets = Dataset.from_dict({
    "anchor": [anchor for anchor, positive, negative in train_triplets],
    "positive": [positive for anchor, positive, negative in train_triplets],
    "negative": [negative for anchor, positive, negative in train_triplets]
})

train_dataset = Dataset.from_dict({
    "sentence": [sent for sent, label in train_set],
    "label": [label for sent, label in train_set]
})

test_dataset = Dataset.from_dict({
    "sentence": [sent for sent, label in test_set],
    "label": [label for sent, label in test_set]
})


print("Data converted into Datasets")

test_sentences = [element[0] for element in test_set]
test_topics = [element[1] for element in test_set]
labeled_sentences = [x[0] for x in labeled_dataset]
labeled_dataset_topics = [x[1] for x in labeled_dataset]


results = {
    "train_triplets": train_triplets,
    "test_triplets": test_triplets,
    "train_dataset_triplets": train_dataset_triplets,
    "train_dataset": train_dataset,
    "test_dataset": test_dataset,
    "test_sentences": test_sentences,
    "test_topics": test_topics,
    "labeled_sentences": labeled_sentences,
    "labeled_dataset_topics": labeled_dataset_topics
}

with open(result_loc, "wb") as file:
    pickle.dump(results, file)
