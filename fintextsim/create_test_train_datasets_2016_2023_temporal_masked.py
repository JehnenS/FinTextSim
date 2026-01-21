import pandas as pd
import numpy as np
import pickle
import os

os.chdir("/mnt/sdb1/home/simonj")

result_loc = "paper1/Data/train_test_sets/train_test_sets_2016_2023_masked.pkl"

#------------Load labeled dataset
dataset_loc = "paper1/Data/Labeled_Dataset/labeled_dataset_2016_2023.pkl"

with open(dataset_loc, "rb") as file:
    ld = pickle.load(file)

ld_results = ld["dataset"]
labeled_dataset = ld_results["labeled_dataset"]
ld_metadata = ld_results["metadata"]


from fintextsim.utils_fintextsim import prepare_triplet_data, year_based_split
from labeled_dataset.utils_labeled_dataset import label_to_keywords, label_to_blacklist

#create time-based train and test-sets
train_set, test_set = year_based_split(labeled_dataset, ld_metadata, year_test_start = 2022)

#apply regularization to the train-set by randomly masking keywords
from fintextsim.Masker import Masker
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#define tokenizer using the FinTextSim base model
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

#initialize masker on train_set
masker = Masker(
    labeled_dataset=train_set,
    label_to_keywords=label_to_keywords,
    tokenizer=tokenizer,
    label_to_blacklist=label_to_blacklist,
)


masked_train_set = masker.get_masked_data(mask_prob = 0.5, use_mask_token = True)
unmasked_train_set = masker.get_masked_data(mask_prob = 0.0, use_mask_token = False)

print(masked_train_set[:5])

masker_test = Masker(
    labeled_dataset=test_set,
    label_to_keywords=label_to_keywords,
    tokenizer=tokenizer,
    label_to_blacklist=label_to_blacklist,
)

masked_test_set = masker_test.get_masked_data(mask_prob = 0.5, use_mask_token = True)
unmasked_test_set = masker_test.get_masked_data(mask_prob = 0.0, use_mask_token = False)


#create triplets
train_triplets = prepare_triplet_data(masked_train_set) #based on masked set
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

train_dataset_masked = Dataset.from_dict({
    "sentence": [sent for sent, label in masked_train_set],
    "label": [label for sent, label in masked_train_set]
})

test_dataset = Dataset.from_dict({
    "sentence": [sent for sent, label in test_set],
    "label": [label for sent, label in test_set]
})

test_dataset_masked = Dataset.from_dict({
    "sentence": [sent for sent, label in masked_test_set],
    "label": [label for sent, label in masked_test_set]
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
    "train_dataset_masked": train_dataset_masked,
    "train_dataset": train_dataset,
    "test_dataset": test_dataset,
    "test_dataset_masked": test_dataset_masked,
    "test_sentences": test_sentences,
    "test_topics": test_topics,
    "labeled_sentences": labeled_sentences,
    "labeled_dataset_topics": labeled_dataset_topics,
    "masked_test_set": masked_test_set,
    "unmasked_test_set": unmasked_test_set,
    "masked_train_set": masked_train_set,
    "unmasked_train_set": unmasked_train_set
}

with open(result_loc, "wb") as file:
    pickle.dump(results, file)

print(f"Results saved to {result_loc}")
