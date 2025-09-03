import pandas as pd
import numpy
import pickle
import os

os.chdir("/mnt/sdb1/home/simonj") #set working directory

#---------Load ciks
file = "Data/"
file_name = "sp500"
full_name = f"{file}{file_name}.pkl"

with open(full_name, "rb") as file:
    sp500_data = pickle.load(file)

# Access the variables within the loaded dictionary
cik_list = sp500_data["rel_cik_full"]

#--------------------------------Load documents

from documents.DocLoader import DocLoader

loader = DocLoader(
    rel_cik_list = cik_list, #run extraction only for the S&P 500 companies (rel ciks)
    zip_path = "10-K",
    keyword = "_10-K_",
    output_dir = "paper1/Data/10-K/"
)

loader.run_extraction()