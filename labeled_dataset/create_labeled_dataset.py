import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os 

os.chdir("/mnt/sdb1/home/simonj") #set working directory

from labeled_dataset.utils_labeled_dataset import topic_names, keywords, exclusion_dict, keyword_blacklist_substring


#----------Load documents and metadata
file_loc = "paper1/Data/10-K/item7/item7_text_outlier_sentences_clean.pkl"

with open(file_loc, "rb") as file:
    result_dict = pickle.load(file)

sentences = result_dict["item7_texts"]
metadata = result_dict["item7_metadata"]

print(f"Number of documents: {len(sentences)}")
print(f"Number of metadata: {len(metadata)}")


######## Define loc where results should be saved
result_loc = "paper1/Data/Labeled_Dataset/labeled_dataset.pkl"
print(f"Result loc: {result_loc}")
#########




#----------------------------
from labeled_dataset.LabeledDatasetCreator import LabeledDatasetCreator

creator = LabeledDatasetCreator(
    sentences = sentences,
    metadata = metadata,
    topic_names = topic_names,
    keyword_list = keywords,
    keyword_blacklist = keyword_blacklist_substring,
    exclusion_dict = exclusion_dict
)

ld_results = creator.run()


#--------save results
with open (result_loc, "wb") as file:
    pickle.dump(ld_results, file)

print(f"Labeled dataset saved to: {result_loc}")