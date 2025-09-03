import torch
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

#-------------------training
def create_test_dataset(dataset, test_ratio = 0.2):
    """
    create a test- and train dataset
    Split by topic to ensure that there is a reasonable amount of examples for all topics in both train- and testset
    """
    # Group the dataset by topic
    topic_groups = {}
    for sentence, topic in dataset:
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append((sentence, topic))
    
    # Split each topic group into train and test sets
    train_dataset = []
    test_dataset = []
    
    for topic, topic_data in topic_groups.items():
        # Split the topic group into train and test sets
        train_sentences, test_sentences = train_test_split([data[0] for data in topic_data], test_size=test_ratio, random_state=42)
        
        # Assign topic labels to train and test sets
        train_topic_labels = [topic] * len(train_sentences)
        test_topic_labels = [topic] * len(test_sentences)
        
        # Combine the sentences and labels into train and test datasets
        train_dataset.extend(zip(train_sentences, train_topic_labels))
        test_dataset.extend(zip(test_sentences, test_topic_labels))

    print("Train- and test dataset created")
    print(f"Number of training sentences: {len(train_dataset)}")
    print(f"Number of test sentences: {len(test_dataset)}")

    return train_dataset, test_dataset


from collections import defaultdict
from random import choice

def prepare_triplet_data(dataset):
    """
    Prepare triplet data
    """
    # Step 1: Separate dataset into sentences and topic labels
    sentences, topics = zip(*dataset)
    
    # Step 2: Group sentences by topic label
    topic_sentences = defaultdict(list)
    for sentence, topic in dataset:
        topic_sentences[topic].append(sentence)
    
    triplet_data = []
    
    # Step 3: Generate triplet data
    for anchor_topic, anchor_sentences in tqdm(topic_sentences.items(), desc = "Progress triplet creation"):
        for anchor_sentence in anchor_sentences:
            # Positive: Choose a sentence from the same topic
            positive_sentence = choice(topic_sentences[anchor_topic])
            # Negative: Choose a sentence from a different topic
            negative_topic = anchor_topic
            while negative_topic == anchor_topic:
                negative_topic = choice(list(topic_sentences.keys()))
            negative_sentence = choice(topic_sentences[negative_topic])
            
            # Step 4: Format the data into InputExample objects
            triplet_data.append((anchor_sentence, positive_sentence, negative_sentence))
            
    print("Triplets prepared")
    return triplet_data

#----------------evaluation
