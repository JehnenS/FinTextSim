import datapath
import gensim
from gensim.models import Nmf
import pickle
import os
os.chdir("/mnt/sdb1/home/simonj") #set working directory

#define path to save topic models
saving_path = "paper1/Topic_Models/NMF/nmf_"

from labeled_dataset.utils_labeled_dataset import keywords

num_topics = len(keywords) #align with the number of topics from the keyword list

#---------------load corpus and dictionary
loc = "paper1/Data/topic_model_input/classical_input_2016_2023.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)

corpus = data["corpus"]
id2word = data["id2word"]
corpus_tfidf = data["corpus_tfidf"]

print("Corpus and id2word loaded")

#-------------------------tf model
print("Fit NMF tf model")
nmf_model = Nmf(
    corpus = corpus,
    id2word = id2word,
    num_topics = num_topics,
    random_state = 42,
    minimum_probability = 0
)

print("NMF tf created")

#-------------------------tfidf model
print("Fit NMF tfidf model")
nmf_model_tfidf = Nmf(
    corpus = corpus_tfidf,
    id2word = id2word,
    num_topics = num_topics,
    random_state = 42,
    minimum_probability = 0
)
print("NMF tfidf created")


#------------------save results

nmf_model.save(f"{saving_path}_tf")
nmf_model_tfidf.save(f"{saving_path}_tfidf")