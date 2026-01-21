import pickle
import gensim
import numpy as np
from gensim.test.utils import datapath
import gensim
from tqdm import tqdm
from bertopic import BERTopic

import os
os.chdir("/mnt/sdb1/home/simonj/") #set wd
#---------------load corpus and dictionary for classical models
loc = "paper1/Data/topic_model_input/classical_input_2016_2023.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)

corpus = data["corpus"]
id2word = data["id2word"]
corpus_tfidf = data["corpus_tfidf"]
texts = data["texts"]
print("Classical Corpus and id2word loaded\n")


text_length_classical = np.mean([len(text) for text in texts])

#---------Load coherence parameters
loc = "paper1/Data/topic_model_input/coherence_data_bertopic_2016_2023.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)

id2word = data["id2word"]
tokens = data["tokens"]
lemmatized_tokens = data["lemmatized_tokens"] 
#id2word_lemmatized = data["id2word_lemmatized"]
print("BERTopic id2word and tokens loaded.\n")

id2word_lemmatized = gensim.corpora.Dictionary(lemmatized_tokens)
print("Lemmatized id2word created")

#----------Load texts
loc = "paper1/Data/topic_model_input/bertopic_input_2016_2023.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)


texts = data["texts"]
print("Texts loaded")

from bertopic_models.utils_bertopic import extract_coherence_parameters



#----------------define the window size for BERTopic approaches
c_window_size_classical = 10
text_length_bertopic = np.mean([len(text) for text in lemmatized_tokens])
print(f"Mean text length classical approaches: {text_length_classical}")
print(f"Mean text length BERTopic approaches: {text_length_bertopic}")
print(f"Window size classical approaches: {c_window_size_classical}")
c_window_size_bertopic = int(round(text_length_bertopic / text_length_classical * c_window_size_classical))
print(f"Window size for BERTopic approaches to guarantee identical coverage: {c_window_size_bertopic}\n")


result_dir = "paper1/Results/BERTopic_Models/bertopic_results_classifier.pkl"
result_dict = {}
result_dict["config"] = {
    "n_topic_words": 5,
    "min_words_for_assignment": 2,
    "max_other_topic_words": 1,
    "c_window_size": c_window_size_bertopic
}

result_dict["results"] = {}


#---------Load embeddings
loc = "paper1/Data/topic_model_input/embeddings_classifier_2016_2023.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)


embeddings_distil_roberta = data["distil_roberta"]



#--------

model_locs = [
    "paper1/Topic_Models/BERTopic/bertopic_distil_roberta"
]

model_names = [
    "bertopic_distil_roberta"
]

embeddings = [
    embeddings_distil_roberta
]



from evaluation.BERTopicModelEvaluator import BERTopicModelEvaluator
from labeled_dataset.utils_labeled_dataset import keywords, topic_names


for i, model_loc in tqdm(enumerate(model_locs), desc = "Model Progress"):
    model_name = model_names[i]
    embedding = embeddings[i]
    
    tm = bertopic_acl = BERTopic.load(model_loc)
    print(f"\n---------------Model loaded - {model_name}------------")

    

    
    #define the evaluator - model-specific
    evaluator = BERTopicModelEvaluator(
        bertopic_model = tm,
        embeddings = embedding,
        keyword_list = keywords,
        topic_names = topic_names,
        id2word = id2word_lemmatized,
        texts = lemmatized_tokens,
        num_words = result_dict["config"]["n_topic_words"],
        min_words_for_assignment = result_dict["config"]["min_words_for_assignment"],
        max_other_topic_words = result_dict["config"]["max_other_topic_words"],
        c_window_size = result_dict["config"]["c_window_size"]
    )

    results = evaluator.run(plot_name = model_name)
    
    result_dict["results"][model_name] = results
    
                 
with open (result_dir, "wb") as file:
    pickle.dump(result_dict, file)