import pandas as pd
import pickle
import numpy as np
from bertopic import BERTopic
import os

os.chdir("/mnt/sdb1/home/simonj")

result_loc = "paper1/Results/BERTopic_Models/topic_probabilities_classifier.pkl"

#------------Load texts
loc = "paper1/Data/10-K/item7/item7_text_outlier_sentences_clean_2016_2023.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)

texts = data["item7_texts"]
meta = data["item7_metadata"]

print(f"Number of texts: {len(texts)}")
print(f"Number of meta: {len(meta)}")


#--------------

from sentence_transformers import SentenceTransformer

embedding_model_locs = [
    "paper1/Fintextsim_Models/fintextsim_2016_2023_modern_bert",
    "paper1/Fintextsim_Models/fintextsim_2016_2023_triplet_modern_bert",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

bertopic_model_locs = [
    "paper1/Topic_Models/BERTopic/bertopic_acl_modern_bert",
    "paper1/Topic_Models/BERTopic/bertopic_htl_modern_bert",
    "paper1/Topic_Models/BERTopic/bertopic_AM",
    "paper1/Topic_Models/BERTopic/bertopic_MPNET"
]

model_names = [
    "bertopic_acl",
    "bertopic_htl",
    "AM",
    "MPNET"
]

results = {}

for i, (emb_loc, tm_loc) in enumerate(zip(embedding_model_locs, bertopic_model_locs)):
    print(f"model_loc: {tm_loc}")
    print(f"Sentence_transformer_loc: {emb_loc}")
    print(f"Model name: {model_names[i]}")

    tm = BERTopic.load(tm_loc, embedding_model = SentenceTransformer(emb_loc))
    topic_distributions, topic_token_distributions = tm.approximate_distribution(texts, use_embedding_model = True, batch_size = 100000) 
    
    result_loc_npy = f"paper1/Results/BERTopic_Models/{model_names[i]}_topic_distributions.npy"
    np.save(result_loc_npy, topic_distributions)
    print(f".npy file of probabilities saved to {result_loc_npy}")
    

    results[model_names[i]] = {
        "topic_distributions": topic_distributions,
        #"topic_token_distributions": topic_token_distributions
    }

with open(result_loc, "wb") as file:
    pickle.dump(results, file)

print(f"Results saved to {result_loc}")