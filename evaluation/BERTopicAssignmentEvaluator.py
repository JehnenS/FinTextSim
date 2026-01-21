import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS
import spacy
from matplotlib import pyplot as plt
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class BERTopicAssignmentEvaluator:
    """
    assess the assignments of topics to sentences across multiple BERTopic models
    """
    def __init__(self, 
                 texts, 
                 model_list, 
                 subplot_titles, 
                 keywords, 
                 num_words:int = 5):
        
        self.num_cols = len(model_list)
        self.texts = texts
        self.model_list = model_list
        self.subplot_titles = subplot_titles
        self.keywords = keywords
        self.num_words = num_words


    
    
    def _lemmatize_(self, topic_words):
        """
        function to lemmatize all words --> comparability to classical approaches, remove noise, etc.
        decrease vocabulary size, etc.
        """
        lemmatized_words = []
        for token in topic_words:
            # Check if the token list is not empty
            if token:
                # Check if the token contains multiple words
                if len(token.split()) > 1:
                    # Apply spaCy pipeline to each word separately and lemmatize it
                    lemmatized_token = ' '.join([word.lemma_ for word in nlp(token)])
                    lemmatized_words.append(lemmatized_token)
                else:
                    # Apply spaCy pipeline to the token and lemmatize it
                    lemmatized_words.append(nlp(token)[0].lemma_)
        return lemmatized_words
    


    def plot_wordclouds(self, doc_num, plot_name=None):
        """
        Generate word clouds for a list of word lists and display them in subplots.
    
        Input:
        word_lists: list of lists of words
        subplot_titles: list of titles for each subplot
        num_cols: number of columns in the subplot grid
        plot_name: name of the file to save the plot (optional)
        """
        print(f"Text: {self.texts[doc_num]}")
        # Flatten combined_keywords into a single keyword list
        keyword_list = [word for sublist in self.keywords for word in sublist]
    
        word_list = []
        #extract the topic representations:
        for model in self.model_list:
            topic = model.topics_[doc_num]
            topic_rep = [word for word, prob in model.get_topic(topic)][:self.num_words]
            word_list.append(topic_rep)
    
        lemmatized_word_list = [self._lemmatize_(topic) for topic in word_list]
    
        # Combine bigrams with underscores
        combined_word_list = []
        for topic in lemmatized_word_list:
            combined_topic = []
            for word in topic:
                if ' ' in word:
                    combined_topic.append(word.replace(' ', '_'))
                else:
                    combined_topic.append(word)
            combined_word_list.append(combined_topic)
        
        
        num_plots = len(combined_word_list)
        num_rows = (num_plots + self.num_cols - 1) // self.num_cols  # Calculate the number of rows needed
    
        # Define a list of 14 distinct colors --> number of topics
        distinct_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        "#ff474c", #'#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#aec7e8',  # light blue
        '#ffbb78',  # light orange
        '#98df8a',  # light green
        '#ff9896',  # light red
        ]
    
        # Create a mapping from keyword to topic color
        keyword_to_color = {}
        for topic_idx, words in enumerate(self.keywords):
            color = distinct_colors[topic_idx]
            for word in words:
                keyword_to_color[word] = color
    
    
        fig, axs = plt.subplots(num_rows, self.num_cols, figsize=(5 * self.num_cols, 5 * num_rows), squeeze=False)
    
        for i, word_list in enumerate(combined_word_list):
            # Create a color function that uses unique colors for topic keywords and black for others
            def color_func(word, *args, **kwargs):
                parts = word.split('_')
                colors = [keyword_to_color.get(part, "black") for part in parts]
    
                # Check the colors of the parts
                if len(set(colors)) == 1:
                    return colors[0]  # All parts have the same color
                elif 'black' in colors and len(set(colors)) == 2:
                    return next(color for color in colors if color != 'black')  # One part is colored, one is black
                elif len(set(colors)) > 1:
                    return 'darkred'  # Parts have different colors
    
            wordcloud = WordCloud(width=250, height=250, background_color="white",
                                  color_func=color_func).generate_from_frequencies({word: 1 for word in word_list})
            
    
            
            row = i // self.num_cols
            col = i % self.num_cols
            axs[row, col].imshow(wordcloud, interpolation="bilinear")
            axs[row, col].axis("off")
            axs[row, col].set_title(self.subplot_titles[i], fontsize = 26)
    
    
        # Remove empty subplots
        for j in range(i + 1, num_rows * self.num_cols):
            fig.delaxes(axs[j // self.num_cols, j % self.num_cols])
        
        # Adjust the layout to make space for the suptitle
        #plt.subplots_adjust(top=0.88)  # Adjust this value as needed
    
        plt.tight_layout()  # Adjust rect to avoid overlapping with suptitle #rect=[0, 0, 1, 0.96]
        plt.subplots_adjust(top=0.88)  # Adjust this value as needed
    
        
        if plot_name:
            plt_loc = f"paper1/Results/BERTopic_Models/topic_assignments/bertopic_{plot_name}_{doc_num}.jpg"
            plt.savefig(plt_loc)
            print(f"Figure saved to {plt_loc}")
    
        plt.show()

