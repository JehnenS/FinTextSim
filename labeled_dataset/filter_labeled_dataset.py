import pickle
import pandas as pd
import os

os.chdir("/mnt/sdb1/home/simonj")
"""
Filter labeled dataset for years of report between 2016 and 2023
"""

result_loc = "paper1/Data/Labeled_Dataset/labeled_dataset_2016_2023.pkl"

#-----------Load full labeled dataset
file_loc = "paper1/Data/Labeled_Dataset/labeled_dataset.pkl"

with open(file_loc, "rb") as file:
    ld_results = pickle.load(file)

#labeled_dataset = ld_results["labeled_dataset"]
labeled_dataset_unique = ld_results["labeled_dataset_unique"]
metadata_unique = ld_results["metadata_unique"]
indices_unique = ld_results["indices_unique"]
keywords = ld_results["keywords"]
topic_names = ld_results["topic_names"]
blacklist = ld_results["blacklist"]
exclusion_dict = ld_results["exclusion_dict"]

#print(f"Dataset: {len(labeled_dataset)}")
print(f"Dataset unique: {len(labeled_dataset_unique)}")

#---------Filter dataset

start_year, end_year = 2016, 2023

filtered_labeled_dataset, filtered_metadata, filtered_indices = zip(*[
    (t, m, i) for t, m, i in zip(labeled_dataset_unique, metadata_unique, indices_unique)
    if m["year_of_report"].isdigit() and start_year <= int(m["year_of_report"]) <= end_year
])
print(f"Filtered Dataset unique: {len(filtered_labeled_dataset)}")


#-----------Save results
results = {}
results["config"] = {
    "keywords": keywords,
    "topic_names": topic_names,
    "blacklist": blacklist,
    "exclusion_dict": exclusion_dict
}


results["dataset"] = {
    "labeled_dataset": filtered_labeled_dataset,
    "metadata": filtered_metadata,
    "indices": filtered_indices
}

with open(result_loc, "wb") as file:
    pickle.dump(results, file)
