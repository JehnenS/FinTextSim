import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import datapath
import os
import gensim
import bertopic
from matplotlib import pyplot as plt
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


os.chdir("/mnt/sdb1/home/simonj")


#---------------Load texts and metadata

loc = "paper1/Data/10-K/item7/item7_text_outlier_sentences_clean_2016_2023.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)

texts = data["item7_texts"]
meta = data["item7_metadata"]

print(f"Number of texts: {len(texts)}")
print(f"Number of meta: {len(meta)}")



#-----------load S&P500 ticker list
#-------load cik-based data
loc = "paper2/Data/FMP/fmp_data_cik.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)

results_cik = data["results"]
results_economic = results_cik["economic"] #extract economic data to make it cleanly ticker-based
results_cik.pop("economic", None) #remove economic from results to make it cleanly ticker-based
print(f"Number of keys from ticker-based results: {len(results_cik.keys())}")
print(f"Number of keys from economic results: {len(results_economic.keys())}")

#--------generate cik-list
from feature_creation.CIKMainSymbolExtractor import CIKMainSymbolExtractor

c_extractor = CIKMainSymbolExtractor(results_cik)
cik_ticker_df = c_extractor.extract_symbols()

# CIK → ticker mapping
cik_to_ticker = (
    cik_ticker_df.groupby("cik")["ticker"]
      .first()   #get first value --> no duplicates, so it should be no issue in general
      .to_dict()
)


#-------------BERTopic models
topic_vector_loc = "paper1/Results/BERTopic_Models/topic_probabilities.pkl"

with open(topic_vector_loc, "rb") as file:
    data = pickle.load(file)

topic_vectors_acl = data["bertopic_acl"]["topic_distributions"]
topic_vectors_htl = data["bertopic_htl"]["topic_distributions"]
topic_vectors_am = data["AM"]["topic_distributions"]
topic_vectors_mpnet = data["MPNET"]["topic_distributions"]

model_names = [
    "fintextsim_acl", 
    "fintextsim_htl", 
    "fintextsim_htl_masked", 
    "am", 
    "mpnet"
]

topic_vectors = [
    topic_vectors_acl, 
    topic_vectors_htl, 
    topic_vectors_htl_masked, 
    topic_vectors_am, 
    topic_vectors_mpnet
]

from feature_creation.TextFeatureCreator import TextFeatureCreator

creator = TextFeatureCreator()


for name, vector in zip(model_names, topic_vectors):
    print(f"\nExtracting topic-distributions and aggregating them into document-topic distributions for {name}")
    result_loc = f"paper1/Data/Features/text_features/text_features_{name}.csv"
    print(f"Results will be saved to {result_loc}")

    doc_vectors = creator.run_bertopic(vector, meta, cik_to_ticker)
    doc_vectors.to_csv(result_loc, index = False)
    print(f"Shape of doc_vectors: {doc_vectors.shape} (incl. identifier cols)")

    print(f"CSV saved to {result_loc}")


#-----------Classical models
#---------------load corpus and dictionary
loc = "paper1/Data/topic_model_input/classical_input_2016_2023.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)

corpus = data["corpus"]
id2word = data["id2word"]
corpus_tfidf = data["corpus_tfidf"]
print("Corpus and id2word loaded")

print(f"Number of documents tf: {len(corpus)}")
print(f"Number of documents tfidf: {len(corpus_tfidf)}")

#----------Load models
from gensim.test.utils import datapath
saving_path = "paper1/Topic_Models/LDA/"

lda_model = gensim.models.ldamodel.LdaModel.load(f"{saving_path}lda_tf")
lda_model_tfidf = gensim.models.ldamodel.LdaModel.load(f"{saving_path}lda_tfidf")

from gensim.test.utils import datapath
saving_path = "paper1/Topic_Models/NMF/"

nmf_model = gensim.models.ldamodel.LdaModel.load(f"{saving_path}nmf_tf")
nmf_model_tfidf = gensim.models.ldamodel.LdaModel.load(f"{saving_path}nmf_tfidf")

#---------Prepare loop
models = [
    lda_model, lda_model_tfidf,
    nmf_model, nmf_model_tfidf
]

corpi = [
    corpus, corpus_tfidf,
    corpus, corpus_tfidf
]

model_names = ["lda_tf", "lda_tfidf", "nmf_tf", "nmf_tfidf"]

#-------------Loop over models
for model, corpus, name in zip(models, corpi, model_names):
    print(f"\nExtracting topic-distributions and aggregating them into document-topic distributions for {name}")
    result_loc = f"paper1/Data/Features/text_features/text_features_{name}.csv"
    print(f"Results will be saved to {result_loc}")
    
    doc_vectors = creator.run_gensim(model, corpus, meta, cik_to_ticker)
    doc_vectors.to_csv(result_loc, index = False)
    print(f"Shape of doc_vectors: {doc_vectors.shape} (incl. identifier cols)")

    print(f"CSV saved to {result_loc}")

