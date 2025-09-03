import os
os.chdir("/mnt/sdb1/home/simonj") #set working directory

from documents.ClassicalModelPreprocessor import ClassicalModelPreprocessor

"""
Run the preprocessing steps for classical models
"""

text_loc = "paper1/Data/10-K/item7/item7_text_outlier_sentences_clean.pkl"
stopwords_loc = "LM_stopwords"
save_loc = "paper1/Data/topic_model_input/classical_input_2016_2023.pkl"

start_year = 2016
end_year = 2023

prep = ClassicalModelPreprocessor(
    text_loc = text_loc,
    stopwords_loc = stopwords_loc,
    save_loc = save_loc,
    start_year = start_year,
    end_year = end_year
)

prep.run_batches()