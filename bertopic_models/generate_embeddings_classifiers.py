import pickle
import os
from tqdm import tqdm
import numpy as np

os.chdir("/mnt/sdb1/home/simonj")

save_loc = "paper1/Data/topic_model_input/embeddings_classifier_2016_2023.pkl"
print(f"Results will be saved to {save_loc}")
#---------Load texts
text_loc = "paper1/Data/topic_model_input/bertopic_input_2016_2023.pkl"
with open(text_loc, "rb") as file:
    data = pickle.load(file)

texts = data["texts"]
print(f"Number of texts: {len(texts)}")


#-------------define model locs and names
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#add model name for saving later on --> no backslashes, etc.
model_name = "distil_roberta"


#-------------------Create embeddings
from bertopic_models.utils_bertopic import generate_embeddings_classifier

all_embeddings_matrix = generate_embeddings_classifier(model, tokenizer, texts, "roberta")


#create dictionary to store results
results = {}

results[model_name] = all_embeddings_matrix


#----------Save results
with open(save_loc, "wb") as file:
    pickle.dump(results, file)

print(f"Results saved to {save_loc}")