import bertopic
from bertopic import BERTopic
from tqdm import tqdm
from gensim import corpora
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def extract_coherence_parameters(model, docs):
    """
    Extract coherence parameters from the BERTopic model

    Args:
        model: BERTopic model
        docs: original documents plugged into the BERTopic model
    """
    #extract vectorizer and analyzer from BERTopic model
    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    #extract features for topic coherence evaluation
    words = vectorizer.get_feature_names_out()
    cleaned_docs = model._preprocess_text(docs)
    tokens = [analyzer(doc) for doc in tqdm(cleaned_docs, desc = "Extracting tokens")]
    print("Fit dictionary.")
    dictionary = corpora.Dictionary(tokens)
    print("Dictionary successfully created.")
    #corpus = [dictionary.doc2bow(token) for token in tokens] --> not necessary for npmi coherence

    return dictionary, tokens


def lemmatize_token_lists(token_lists, batch_size: int = 500, n_process: int = -1):
    """
    Lemmatize a list of tokenized documents.
    If there are multi-word-tokens, lemmatize each of them separately but keep the token itself instead of splitting it into single words
    
    Args:
        token_lists (list[list[str]]): List of tokenized documents (tokens may contain multiple words).
        batch_size (int): Batch size for spaCy processing.
        n_process (int): Number of processes to use (-1 = all cores).
    
    Returns:
        list[list[str]]: List of lemmatized token lists.
    """
    lemmatized_docs = []

    #iterate over each tokenized sentence
    for tokens in tqdm(token_lists, desc = "Progress"):
        lemmatized_tokens = [] #list to store results per sentence
        for token in tokens: #iterate over all tokens within sentence
            if not token:
                continue
            #multi-word token
            if len(token.split()) > 1:
                doc = nlp(token)
                lemmatized = " ".join([w.lemma_ for w in doc])
                lemmatized_tokens.append(lemmatized)
            else:
                #single token
                doc = nlp(token)
                lemmatized_tokens.append(doc[0].lemma_)
        lemmatized_docs.append(lemmatized_tokens)
    
    return lemmatized_docs
