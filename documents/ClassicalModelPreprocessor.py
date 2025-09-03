import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
import gensim
from gensim import corpora
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class ClassicalModelPreprocessor:
    """
    Transform the processed texts into the format necessary for classical gensim topic models
    """
    def __init__(self, text_loc, stopwords_loc, save_loc, start_year:int = 2016, end_year:int = 2023):
        self.text_loc = text_loc
        self.stopwords_loc = stopwords_loc
        self.save_loc = save_loc
        self.start_year = start_year
        self.end_year = end_year

        self.texts = None
        self.metadata = None
        self.stopwords_list = None
        self.corpus = None
        self.corpus_tfidf = None
        self.id2word = None

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

    def _load_lm_stopwords_(self):
        """
        Load LM stopwords from folder
        """
        stopwords_list = []
        for root, _, files in os.walk(self.stopwords_loc):
            for filename in files:
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="latin-1") as file:
                    doc_content = file.read()
                    doc_content = doc_content.replace("\n", " ").lower()
                    tokens = self._gensim_prep_(doc_content, use_stopwords = False)  
                    stopwords_list.extend(tokens)
    
        exclude_on_top = ["mr", "mrs", "ms"]
        remove_from_lm_stopwords = ["sale", "cash", "brand"]
    
        lm_stopwords = stopwords_list + exclude_on_top
        lm_stopwords = [w for w in lm_stopwords if w not in remove_from_lm_stopwords]

        print(lm_stopwords[:10])
        print(f"Number of stopwords: {len(lm_stopwords)}\n")
    
        self.stopwords_list = lm_stopwords


    def _remove_stopwords_(self, text):
        """
        Remove stopwords from text pre tokenization
    
        text: document
        """
        tokens = text.split() #split text into words/by whitespaces
        filtered_tokens = [token for token in tokens if token not in self.stopwords_list]
            
        text_filtered = " ".join(filtered_tokens) #join words back into text
    
        return text_filtered

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
        
    def _lemmatize_batch_(self, texts, batch_size: int = 500, n_process: int = -1):
        """
        Lemmatize a list of texts using spaCy with batching & multiprocessing.
        
        Args:
            texts (list[str]): List of documents (strings).
            batch_size (int): Number of docs to process per batch.
            n_process (int): Number of processes (use -1 for all cores).
        
        Returns:
            list[str]: List of lemmatized texts.
        """
        lemmatized_texts = []
        for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process),
                        total=len(texts), desc="Lemmatization"):
            tokens = [token.lemma_ for token in doc]
            lemmatized_texts.append(" ".join(tokens))
        
        return lemmatized_texts


    def _gensim_prep_(self, text, use_stopwords = True):
        """
        Function to perform simple gensim preprocessing: tokenization, removing accents, punctuation, etc.
        
        input: 
        text: text/sentence
        stopwords_list: stopwords list to remove stopwords post tokenization
        """
        final = []

        tokens = gensim.utils.simple_preprocess(text, deacc = True) #tokenize and preprocess
        if use_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords_list]

        return tokens

    def _generate_phrases_(self, texts, min_count:int, threshold:int = 80):
        """
        Helper function in order to create bigrams and trigrams
        Generate phrases
    
        Input:
        texts: list of lists containing tokens per document
        min_count (int): number of times the bigram/trigram has to occur
        threshold: variable for creating bigrams/trigrams
        """
        bigram_phrases = gensim.models.Phrases(texts, min_count = min_count, threshold = threshold)
        trigram_phrases = gensim.models.Phrases(bigram_phrases[texts], min_count = min_count, threshold = threshold)
    
        return bigram_phrases, trigram_phrases

    def _create_bigram_trigram_models_(self, bigram_phrases, trigram_phrases):
        """
        Helper function on order to create bigrams/trigrams
        Create bigram/trigram models
    
        Input:
        output from generate_phrases function
        """
        bigram_model = gensim.models.phrases.Phraser(bigram_phrases)
        trigram_model = gensim.models.phrases.Phraser(trigram_phrases) 
        
        return bigram_model, trigram_model
    
    
    def _make_bigrams_(self, texts, bigram_model):
        """
        Helper function on order to create bigrams/trigrams
        Make bigrams
    
        Input:
        texts: list of lists containing tokens per document
        bigram_model: output from create_bigram_trigram_models function
        """
        bigrams = [bigram_model[doc] for doc in tqdm(texts, desc = "Bigram creation")]
        return bigrams
    
    def _make_trigrams_(self, texts, bigram_model, trigram_model):
        """
        Helper function on order to create bigrams/trigrams
        Make trigrams
    
        Input:
        texts: bigrams from make_bigrams function
        bigram_model: output from create_bigram_trigram_models function
        trigram_model: output from create_bigram_trigram_models function
        """
        trigrams = [trigram_model[bigram_model[doc]] for doc in tqdm(texts, desc = "Trigram creation")]
        return trigrams
    
    
    def bigrams_trigrams(self, texts, min_count:int, threshold:int = 80):
        """
        wrapper method to create bigrams/trigrams --> combination of the earlier functions
    
        Input: 
        texts: list of lists containing tokens per document
        min_count (int): number of times the bigram/trigram has to occur
        threshold: variable for creating bigrams/trigrams
        """
        #1. Generate phrases
        bigram_phrases, trigram_phrases = self._generate_phrases_(texts, min_count, threshold)
    
        #2. Create bigram/trigram models
        bigram_model, trigram_model = self._create_bigram_trigram_models_(bigram_phrases, trigram_phrases)
    
        #3. Make bigrams/trigrams
        data_bigrams = self._make_bigrams_(texts, bigram_model)
        data_bigrams_trigrams = self._make_trigrams_(data_bigrams, bigram_model, trigram_model)
    
        return data_bigrams_trigrams

    def corpus_id2word_creation(self, tokens):
        """
        Create corpus and id2word based on token input
        """
        print("Create id2word")
        #create dictionary
        id2word = corpora.Dictionary(tokens)
        print("Create corpus")
        #create corpus
        corpus = [id2word.doc2bow(doc) for doc in tqdm(tokens, desc = "Creation of corpus")]

        #tfidf corpus
        tfidf = gensim.models.TfidfModel(corpus, id2word)
        corpus_tfidf = tfidf[corpus]
        
        print(f"Number of unique tokens: {len(id2word)}")
        print(f"Number of documents: {len(corpus)}")
        print(f"Number of documents (tfidf): {len(corpus_tfidf)}\n")

        self.corpus = corpus
        self.corpus_tfidf = corpus_tfidf
        self.id2word = id2word

    def _save_(self, final_texts):
        with open(self.save_loc, "wb") as file:
            pickle.dump({
                "texts": final_texts,
                "metadata": self.metadata,
                "stopwords": self.stopwords_list,
                "corpus": self.corpus,
                "corpus_tfidf": self.corpus_tfidf,
                "id2word": self.id2word
            }, file)

    def run(self, share_min_count:float = 0.00005):
        """
        Wrapper method to run preprocessing for classical models

        share_min_count: share in which the bigrams/trigrams have to a appear to be considered a bigram/trigram
        """
        self._load_texts_() #load texts (and metadata)
        self._load_lm_stopwords_() #load stopwords

        #create list to store results
        final_texts = []

        #iterate over each text - remove stopwords, lemmatize texts, tokenize into words, remoce accents, etc.
        for text in tqdm(self.texts, desc = "Progress"):
            text_stopwords_removal = self._remove_stopwords_(text) #remove stopwords pre-tokenization and lemmatization
            text_lemmatized = self._lemmatize_(text_stopwords_removal) #lemmatized the stopword removed texts
            text_final = self._gensim_prep_(text_lemmatized)
            final_texts.append(text_final)

        #create bigrams and trigrams from cleaned text
        text_bigrams_trigrams = self.bigrams_trigrams(final_texts, min_count = int(max(1, len(final_texts) * share_min_count)))

        #create corpus and id2word for bigram/trigram text
        self.corpus_id2word_creation(text_bigrams_trigrams)
        
        #save results
        self._save_(text_bigrams_trigrams)   

    def run_batches(self, batch_size:int = 1024, share_min_count:float = 0.00005):
        """
        Wrapper method running tokenization in batches instead of single documents
        """
        self._load_texts_() #load texts (and metadata)
        self._load_lm_stopwords_() #load stopwords

        
        # 1. remove stopwords first (optional — see earlier suggestion to skip this)
        texts_no_stopwords = [self._remove_stopwords_(text) for text in tqdm(self.texts, desc = "Remove stopwords pre-tokenization")]
        
        # 2. lemmatize in batches
        lemmatized_texts = self._lemmatize_batch_(texts_no_stopwords)
        
        # 3. tokenize with gensim
        final_texts = [self._gensim_prep_(text) for text in tqdm(lemmatized_texts, desc = "Gensim prep")]

        #4. create bigrams and trigrams from cleaned text
        text_bigrams_trigrams = self.bigrams_trigrams(final_texts, min_count = int(max(1, len(final_texts) * share_min_count)))

        #5. create corpus and id2word for bigram/trigram text
        self.corpus_id2word_creation(text_bigrams_trigrams)
        
        #6. save results
        self._save_(text_bigrams_trigrams)