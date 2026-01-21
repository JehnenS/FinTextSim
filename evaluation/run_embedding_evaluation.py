import pickle
import gensim
from bertopic import BERTopic
from matplotlib import pyplot as plt
import cupy as cp
from cuml.metrics import pairwise_distances
import numpy as np

import os

results_dict = {}
results_dict["config"] = {
    "n_neighbors": 125
}
results_dict["Results"] = {}

os.chdir("/mnt/sdb1/home/simonj/")
result_dir = "paper1/Results/Embeddings/embedding_results.pkl"

#-------------load embeddings
from evaluation.EmbeddingEvaluator import EmbeddingEvaluator
from labeled_dataset.utils_labeled_dataset import topic_names


loc = "paper1/Results/Embeddings/test_data/fintextsim_2016_2023_triplet.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)

fts_modernbert = data["training_results"]["modern_bert"]
fts_modernbert_embeddings = fts_modernbert['test_embeddings_fintextsim']
fts_topics = data["training_results"]["modern_bert"]["test_topics"]
fts_sentences = data["training_results"]["modern_bert"]["test_sentences"]
fts_modernbert_embeddings.shape



am_embeddings = data["basics"]["test_embeddings_ots1"]
mpnet_embeddings = data["basics"]["test_embeddings_ots2"]


embeddings = [fts_modernbert_embeddings, am_embeddings, mpnet_embeddings]
model_locs = [
    "paper1/Topic_Models/BERTopic/bertopic_htl_modern_bert",
    "paper1/Topic_Models/BERTopic/bertopic_AM", "paper1/Topic_Models/BERTopic/bertopic_MPNET"
]
model_names = ["FinTextSim", "AM", "MPNET"]



#iterate over all embeddings
for i, embedding in enumerate(embeddings):
    model_name = model_names[i]
    print(f"Model name: {model_name}")

    tm = BERTopic.load(model_locs[i])
    print(f"\nBERTopic Model loaded")

    evaluator = EmbeddingEvaluator(
        embeddings = embedding,
        topics = fts_topics,
        sentences = fts_sentences,
        topic_names = topic_names,
        bertopic_model = tm,
    )

    embedding_results = evaluator.run(fig_name = model_name)
    evaluator.plot(n_neighbors = results_dict["config"].get("n_neighbors"), fig_name = model_name)

    results_dict["Results"][model_name] = embedding_results

                
with open (result_dir, "wb") as file:
    pickle.dump(results_dict, file)

print(f"Results saved to {result_dir}")