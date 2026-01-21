import pickle
import os

os.chdir("/mnt/sdb1/home/simonj")

save_loc = "paper1/Data/topic_model_input/embeddings_2016_2023_temporal.pkl"

#---------Load texts
text_loc = "paper1/Data/topic_model_input/bertopic_input_2016_2023.pkl"
with open(text_loc, "rb") as file:
    data = pickle.load(file)

texts = data["texts"]
print(f"Number of texts: {len(texts)}")


#-------------define model locs and names
from sentence_transformers import SentenceTransformer

model_locs = [
    #"paper1/Fintextsim_Models/fintextsim_2016_2023_temporal_modern_bert",
    "paper1/Fintextsim_Models/fintextsim_2016_2023_triplet_temporal_modern_bert"
    #"sentence-transformers/all-MiniLM-L6-v2", 
    #"sentence-transformers/all-mpnet-base-v2"
]

model_names = [
    #"acl_modern_bert_temporal"
    "htl_temporal"
    #"AM", 
    #"MPNET"
]


#-------------------Create embeddings
#create dictionary to store results
results = {}

#iterate over each model-loc
for i, model_loc in enumerate(model_locs):
    #load sentence transformer
    model = SentenceTransformer(model_loc)

    print(f"Sentence-Transformer loaded - {model_names[i]}")
    embeddings = model.encode(texts, show_progress_bar = True)

    #save results with model name as key
    results[model_names[i]] = embeddings


#----------Save results
with open(save_loc, "wb") as file:
    pickle.dump(results, file)

print(f"Results saved to {save_loc}")