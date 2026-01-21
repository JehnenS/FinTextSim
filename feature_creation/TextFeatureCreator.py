import pandas as pd
import numpy as np
from tqdm import tqdm


class TextFeatureCreator:
    def __init__(self):
        """
        Class to create text features --> based on document-topic distributions
        """

    def extract_topic_distributions_gensim(self, model, corpus):
        """
        Return document-topic probability matrix of shape (n_docs, n_topics)
        """
        num_topics = model.num_topics

        #topic-distribution per document in corpus, i.e. sentence
        topic_vectors = []
    
        for bow in tqdm(corpus, desc="Extract Topic-Distributions"):
            topic_dist = model.get_document_topics(bow, minimum_probability=0)
            #create a vector of zeros to intialize the topic vector
            topic_vector = np.zeros(num_topics, dtype=np.float32)
            #fill values for which we have probabilities
            for topic_id, prob in topic_dist:
                topic_vector[topic_id] = prob  #fill in the probabilities
            topic_vectors.append(topic_vector)
    
        topic_vectors = np.stack(topic_vectors)  # shape = (n_docs, n_topics)
        print(f"Shape of topic vectors: {topic_vectors.shape}")
        return topic_vectors

    def aggregate_topic_distributions(self, topic_vectors, meta):
        """
        Aggregate sentence-level topic information into document-level topic information
        """
        #transform metadata and topic vectors into dataframe
        meta_df = pd.DataFrame(meta)
        topic_vector_df = pd.DataFrame(topic_vectors)

        #concatenate meta df and topic-distributions horizontally --> the correspond to each other, i.e. row 0 from metadata corresponds to row 0 in topic-vectors --> same sentence
        df = pd.concat([meta_df.reset_index(drop=True), topic_vector_df.reset_index(drop=True)], axis=1)

        #only take numeric columns (features)
        feature_cols = topic_vector_df.columns
        
        #group by doc_id and take the mean for each topic column
        doc_vectors = df.groupby("doc_id")[feature_cols].mean().reset_index()
        print(f"Shape of document vectors: {doc_vectors.shape} (incl. doc-id column)")

        return doc_vectors

    def _merge_with_metadata(self, doc_vectors, meta, cik_ticker_mapping):
        """
        Merge textual information with metadata --> ticker and year_of_report in order to link text data with financial predictors
        """
        meta_df = pd.DataFrame(meta)

        #map cik → ticker (vectorized)
        meta_df["ticker"] = meta_df["cik"].map(cik_ticker_mapping)
        
        #clean & format columns
        meta_df = meta_df[["doc_id", "ticker", "filing_date", "cik", "year_of_report"]] #extract only relevant columns
        meta_df["year_of_report"] = pd.to_numeric(meta_df["year_of_report"], errors="coerce") #transform year of report to numeric format
        meta_df = meta_df.dropna(subset=["year_of_report"])  #handle missing years
        meta_df["year_of_report"] = meta_df["year_of_report"].astype(int) #convert to int safely (after dropping NaNs)
        meta_df["filing_date"] = pd.to_datetime(meta_df["filing_date"], format="%Y%m%d", errors = "coerce") #transform to date format
        
        #drop duplicate doc_ids
        meta_df = meta_df.drop_duplicates(subset="doc_id").reset_index(drop=True)

        full_df = doc_vectors.merge(meta_df, on = "doc_id", how = "inner") #merge doc vectors with metadata
        #full_df = full_df.drop(columns = ["doc_id", "filing_date", "cik"]) #drop irrelevant columns --> ticker and year of report as identifier

        return full_df
        
        
    def run_gensim(self, model, corpus, meta, cik_ticker_mapping):
        """
        Wrapper method to run extraction of topic-distributions per sentence and aggregating them into topic-distributions per document
        """
        topic_vectors = self.extract_topic_distributions_gensim(model, corpus)
        doc_vectors = self.aggregate_topic_distributions(topic_vectors, meta)
        doc_vectors_ticker_year = self._merge_with_metadata(doc_vectors, meta, cik_ticker_mapping)

        return doc_vectors_ticker_year

    def run_bertopic(self, topic_vectors, meta, cik_ticker_mapping):
        """
        Wrapper method to create document vectors for BERTopic topic distributions
        """
        doc_vectors = self.aggregate_topic_distributions(topic_vectors, meta)
        doc_vectors_ticker_year = self._merge_with_metadata(doc_vectors, meta, cik_ticker_mapping)

        return doc_vectors_ticker_year