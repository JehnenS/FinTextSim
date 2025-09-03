import pickle
import gensim
import numpy as np
from gensim.test.utils import datapath
import gensim
from tqdm import tqdm

import os


os.chdir("/mnt/sdb1/home/simonj/")
result_dir = "paper1/Results/Classical_Models/classical_results.pkl"
result_dict = {}
result_dict["config"] = {
    "n_topic_words": 5,
    "min_words_for_assignment": 2,
    "max_other_topic_words": 1,
    "c_window_size": 10
}
result_dict["results"] = {}

#---------------load corpus and dictionary
loc = "paper1/Data/topic_model_input/classical_input_2016_2023.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)

corpus = data["corpus"]
id2word = data["id2word"]
corpus_tfidf = data["corpus_tfidf"]
texts = data["texts"]
print("Corpus and id2word loaded")


#--------

model_locs = [
    #"paper1/Topic_Models/LDA/lda_tf", 
    "paper1/Topic_Models/LDA/lda_tfidf",
    #"paper1/Topic_Models/NMF/nmf_tf", 
    "paper1/Topic_Models/NMF/nmf_tfidf"
]

model_names = [
    #"lda_tf", 
    "lda_tfidf",
    #"nmf_tf", 
    "nmf_tfidf"
]

corpi = [corpus, corpus_tfidf, corpus, corpus_tfidf]

from evaluation.ClassicalModelEvaluator import ClassicalModelEvaluator
from labeled_dataset.utils_labeled_dataset import keywords, topic_names

for i, model_loc in tqdm(enumerate(model_locs), desc = "Model Progress"):
    model_name = model_names[i]
    
    tm = gensim.models.ldamodel.LdaModel.load(model_loc)
    print(f"\n-------------Model loaded - {model_name}----------------------")

    #define the evaluator - model-specific
    evaluator = ClassicalModelEvaluator(
        model = tm,
        texts = texts,
        id2word = id2word,
        corpus = corpi[i],
        keywords = keywords,
        topic_names = topic_names,
        n_topic_words = result_dict["config"]["n_topic_words"],
        min_words_for_assignment = result_dict["config"]["min_words_for_assignment"],
        max_other_topic_words = result_dict["config"]["max_other_topic_words"],
        c_window_size = result_dict["config"]["c_window_size"]
    )

    results = evaluator.run(plot_name = model_name)
    
    result_dict["results"][model_name] = results
    
                 
with open (result_dir, "wb") as file:
    pickle.dump(result_dict, file)