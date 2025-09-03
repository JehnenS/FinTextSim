import pickle
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
import os


class BERTopicModelPreprocessor:
    """
    Transform the processed texts into the format necessary for classical gensim topic models
    """
    def __init__(self, text_loc, save_loc, start_year:int = 2016, end_year:int = 2023):
        self.text_loc = text_loc
        self.save_loc = save_loc
        self.start_year = start_year
        self.end_year = end_year

        self.texts = None
        self.metadata = None
        self.texts_lemmatized = None
        


    def _load_texts_(self):
        """
        Load the processed and sentence-tokenized texts and metadata (equal to BERTopic approach)
        """
        with open(self.text_loc, "rb") as file:
            result_dict = pickle.load(file)
        
        sentences = result_dict["item7_texts"]
        metadata = result_dict["item7_metadata"]
        
        print(f"Number of documents: {len(sentences)}")
        print(f"Number of metadata: {len(metadata)}")
        print("Texts loaded.\n")

        #filter for years between 
        filtered_texts, filtered_metadata = zip(*[
            (t, m) for t, m in zip(sentences, metadata)
            if m["year_of_report"].isdigit() and self.start_year <= int(m["year_of_report"]) <= self.end_year
        ])

        print(f"Number of filtered documents: {len(filtered_texts)}")
        print(f"Number of filtered metadata: {len(filtered_metadata)}\n")

        self.texts = filtered_texts
        self.metadata = filtered_metadata

    def _lemmatize_(self, text):
        """
        function to lemmatize the text
        """
        #cast spacy document for text
        doc = nlp(text)
    
        #create list to store results
        text_lemma = []
        
        #iterate through each token of the document and check pos + keep token if it is in relevant pos list
        for token in doc:     
            text_lemma.append(token.lemma_)
    
        #cast tokens back to text
        final_text_lemma = " ".join(text_lemma)

        return final_text_lemma


  

    def _save_(self):
        """
        Save texts, metadata and lemmatized texts
        """
        with open(self.save_loc, "wb") as file:
            pickle.dump({
                "texts": self.texts,
                "metadata": self.metadata,
                "texts_lemmatized": self.texts_lemmatized
            }, file)

    def run(self):
        """
        Wrapper method to run preprocessing for classical models
        """
        self._load_texts_() #load texts (and metadata)

        #create list to store results
        texts_lemmatized = []

        #iterate over each text
        for text in tqdm(self.texts, desc = "Progress"):
            text_lemmatized = self._lemmatize_(text) #lemmatized the stopword removed texts
            texts_lemmatized.append(text_lemmatized)

        self.texts_lemmatized = texts_lemmatized

        self._save_()