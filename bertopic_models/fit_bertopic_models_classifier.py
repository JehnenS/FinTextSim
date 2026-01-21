import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import umap
import hdbscan
import bertopic
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
        
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP

os.chdir("/mnt/sdb1/home/simonj")

#----------Load embeddings

embedding_loc = "paper1/Data/topic_model_input/embeddings_classifier_2016_2023.pkl"

with open(embedding_loc, "rb") as file:
    embedding_data = pickle.load(file)


#--------Load texts
text_loc = "paper1/Data/topic_model_input/bertopic_input_2016_2023.pkl"
with open(text_loc, "rb") as file:
    data = pickle.load(file)

texts = data["texts"]
print(f"Number of texts: {len(texts)}")

#---------------load keywords + stopwords
stopword_loc = "lm_stopwords.pkl"  
with open(stopword_loc, "rb") as file:
    data = pickle.load(file)
        
        # Access the variables within the loaded dictionary
lm_stopwords = data["lm_stopwords"]        


#-----------

from labeled_dataset.utils_labeled_dataset import keywords

############Define the configuration of training
config_dict = {
    #UMAP
    "n_components": 10, #refers to dimensionality of embeddings after reducing --> tradeoff between maximizing information kept in embeddings while also reducing as much as possible (started with 5)
    "n_neighbors": 125, #increasing leads to more global instead of local topics
    "min_dist": 0.0,
    "metric_umap": "cosine",
    "unique" : True,
    "low_memory": True,
    "random_state": 42,
    #HDBSCAN
    "min_cluster_size" : 5000,
    "min_samples": 50,
    "metric_hdbscan": "euclidean",
    "cluster_selection_method": "eom",
    "gen_min_span_tree": True,
    "prediction_data": True,
    #ctfidf
    "bm25_weighting": True,
    "reduce_frequent_words": True,
    "seed_words": keywords,
    "seed_multiplier": 50,
    #vectorizer
    "stop_words": lm_stopwords,
    #"min_df": 50,
    "ngram_range": (1, 2),
}

config_save_loc = "paper1/Topic_Models/BERTopic/config_classifiers.pkl"
##############


import os
os.getcwd()

os.chdir("/mnt/sdb1/home/simonj/")



##------------------Define the base models within BERTopic

umap_model = umap.UMAP(
    n_neighbors =  config_dict.get("n_neighbors"), #increasing leads to more global instead of local topics --> desirable (started with 5)
    n_components = config_dict.get("n_components"), #refers to dimensionality of embeddings after reducing --> tradeoff between maximizing information kept in embeddings while also reducing as much as possible (started with 5)
    min_dist = config_dict.get("min_dist"),
    random_state = config_dict.get("random_state"),
    metric = config_dict.get("metric_umap"), #change to cosine
    unique = config_dict.get("unique"),
    low_memory = config_dict.get("low_memory")
)


hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size = config_dict.get("min_cluster_size"), #minimum of words in a cluster (started with 5)
    min_samples = config_dict.get("min_samples"), #controls number of outliers, standard: equal to min_cluster_size, if setting significantly lower: maybe reducing amount of noise
    metric = config_dict.get("metric_hdbscan"),  #after umap: low-dimensional data so that not much optimization is necessary --> change when using high dimensionality in n_components in UMAP
    cluster_selection_method = config_dict.get("cluster_selection_method"), 
    gen_min_span_tree = config_dict.get("gen_min_span_tree"),
    prediction_data = config_dict.get("prediction_data")
)

#tune ctidf model        
ctfidf_model = ClassTfidfTransformer(
    bm25_weighting = config_dict.get("bm25_weighting"), 
    reduce_frequent_words = config_dict.get("reduce_frequent_words"),
    seed_words = config_dict.get("seed_words"), #use the combined keywords as seed words
    seed_multiplier = config_dict.get("seed_multiplier")
) 
        
        
#tune the vectorizer
vectorizer_model = CountVectorizer(
    stop_words = config_dict.get("stop_words"), 
    ngram_range = config_dict.get("ngram_range")
)



#---------------Loop over the embeddings/models
for model in embedding_data.keys():
    #extract embeddings
    embeddings = embedding_data[model]
    print(f"Model name: {model}")
    print(f"Shape of embeddings: {embeddings.shape}")
    model_loc = f"paper1/Topic_Models/BERTopic/bertopic_{model}"
    print(f"Model will be saved to {model_loc}")


    #define BERTopic model
    bertopic_model = BERTopic(
        language = "english",
        top_n_words = 10, #number of words per topic that want to be extracted --> keep below 30 and preferably between 10 and 20
        nr_topics = "auto", #specifies number of topics after training the model --> e.g. if model results in 100 topics, but nr_topics set to 20: reduce number of topics from 100 to 20 
        calculate_probabilities = True, #set to False to increase the speed --> probs can be approximated later on
        umap_model = umap_model, #umap_model_cuml,  #perform dimensionality reduction
        hdbscan_model = hdbscan_model, #hdbscan_model_cuml,  #cluster reduced embeddings into similar groups --> create topics
        vectorizer_model = vectorizer_model, #create topic representations
        ctfidf_model = ctfidf_model #get accurate representation from bag-of-words matrix --> adjustment of tf-idf to work on topic-level instead of document level
        #representation_model = representation_model
    )

    
        

    print("Fit BERTopic model")
    topics, probs = bertopic_model.fit_transform(texts, embeddings)
        
    print("TM ready")
    bertopic_model.save(model_loc, serialization="safetensors", save_ctfidf=True)
    print(f"Model saved to {model_loc}")
    


with open(config_save_loc, "wb") as file:
    pickle.dump(config_dict, file)
    
print(f"Config file saved to {config_save_loc}")