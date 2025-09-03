import pickle
from bertopic_models.utils_bertopic import extract_coherence_parameters, lemmatize_token_lists
from bertopic import BERTopic
import gensim
import os

os.chdir("/mnt/sdb1/home/simonj")

#Load BERTopic model --> for all models, we have the same input and underlying models (particularly vectorizer and stopwords) --> hence, it is sufficient to extract id2word, tokens and lemmatize the tokens only once and use across all models

save_loc = "paper1/Data/topic_model_input/coherence_data_bertopic_2016_2023.pkl"

#----------load bertopic model
bertopic_htl = BERTopic.load("paper1/Topic_Models/BERTopic/bertopic_acl_modern_bert")

#----------Load texts
loc = "paper1/Data/topic_model_input/bertopic_input_2016_2023.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)


texts = data["texts"]

#---------Processing
id2word, tokens = extract_coherence_parameters(bertopic_htl, texts)
lemmatized_tokens = lemmatize_token_lists(tokens, batch_size = 4096)
id2word_lemmatized = gensim.corpora.Dictionary(lemmatized_tokens)

#------Save results

result_dict = {
    "id2word": id2word,
    "tokens": tokens,
    "lemmatized_tokens": lemmatized_tokens,
    "id2word_lemmatized": id2word_lemmatized
}

with open(save_loc, "wb") as file:
    pickle.dump(result_dict, file)

print(f"Output saved to {save_loc}")