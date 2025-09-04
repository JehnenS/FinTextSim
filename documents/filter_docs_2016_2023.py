import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

os.chdir("/mnt/sdb1/home/simonj")


#-----------------load texts and metadata
loc = "paper1/Data/10-K/item7/item7_text_outlier_sentences_clean.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)

texts_sentences = data["item7_texts"]
meta_sentences = data["item7_metadata"]


#----------------Filter based on year of report
start_year, end_year = 2016, 2023

filtered_texts_final, filtered_metadata_final = zip(*[
    (t, m) for t, m in zip(texts_sentences, meta_sentences)
    if m["year_of_report"].isdigit() and start_year <= int(m["year_of_report"]) <= end_year
])
print(f"Filtered Dataset for texts between 2016 and 2023: {len(filtered_texts_final)}")


#-------------Save results

file_loc = f"paper1/Data/10-K/item7/item7_text_outlier_sentences_clean_{start_year}_{end_year}.pkl"

with open(file_loc, "wb") as file:
    pickle.dump(
        {
            "item7_texts": filtered_texts_final,
            "item7_metadata": filtered_metadata_final
        }, file
    )

print(f"Results for years of report between {start_year} and {end_year} saved to {file_loc}.")
