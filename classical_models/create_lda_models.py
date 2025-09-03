import datapath
import gensim
import pickle
import os
os.chdir("/mnt/sdb1/home/simonj") #set working directory

#define path to save topic models
saving_path = "paper1/Topic_Models/LDA/lda_"

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
print("Fit LDA tf model")
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics= num_topics,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
    eta = "auto",
    random_state = 42
)

print("LDA tf created")

#-------------------------tfidf model
print("Fit LDA tfidf model")
lda_model_tfidf = gensim.models.ldamodel.LdaModel(
    corpus=corpus_tfidf,
    id2word=id2word,
    num_topics= num_topics,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
    eta = "auto",
    random_state = 42
)

print("LDA tfidf created")


#------------------save results

lda_model.save(f"{saving_path}_tf")
lda_model_tfidf.save(f"{saving_path}_tfidf")
