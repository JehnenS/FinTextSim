import pickle
import gensim
from bertopic import BERTopic
from matplotlib import pyplot as plt
import cupy as cp
from cuml.metrics import pairwise_distances
import numpy as np

import os

import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
#---------------
from gensim.models import CoherenceModel

#-------------------topic precision
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import numpy as np
from collections import Counter




class BERTopicModelEvaluator:
    """
    Evaluate topic models (BERTopic) based on coherence, intra- and intertopic similarity, topic-precision
    """
    def __init__(self,
                 bertopic_model,
                 embeddings,
                 keyword_list,
                 topic_names,
                 id2word,
                 texts,
                 min_words_for_assignment:int = 2,
                 max_other_topic_words:int = 1,
                 num_words:int = 5,
                c_window_size:int = 10):
        
        self.embeddings = embeddings
        self.bertopic_model = bertopic_model
        self.keyword_list = keyword_list
        self.topic_names = topic_names
        self.id2word = id2word
        self.texts = texts
        self.min_words_for_assignment = min_words_for_assignment
        self.max_other_topic_words = max_other_topic_words
        self.num_words = num_words
        self.c_window_size = c_window_size
        self.embeddings_clean = None
        self.topics_clean = None
        # Extract keywords from model
        self.lemmatized_topic_words = None
        
       

        # Flatten combined_keywords into a single keyword list
        self.keyword_list_flat = [word for sublist in keyword_list for word in sublist]

        
    
    def _extract_topic_words_(self):
        """
        Extract the top-n topic-words for each topic from the BERTopic model (based on ctf-idf score)
        """
        #get number of topics - exclude outlier topic
        topics = self.bertopic_model.get_topic_freq()
        clean_topics = topics[topics["Topic"] != -1]
        num_topics = len(clean_topics)

        #extract and lemmatize topic words
        topic_words = [[words for words, _ in self.bertopic_model.get_topic(topic)] for topic in range(num_topics)] #iterate over each topic
        topic_words_rel = [word[:self.num_words] for word in topic_words] #get top-n-words per topic --> representative words
    
        return topic_words_rel

    def lemmatize_coherence_data(self, tokens):
        """
        lemmatize all words --> comparability to classical approaches, remove noise, etc.
        decrease vocabulary size, etc.
        """
        lemmatized_words = []
        for token in tokens:
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

    def npmi_coherence(self, plot_name = None):
        """
        Compute coherence for BERTopic model

        dictionary: dictionary from the corpus
        tokens: list of lists of tokenized (and lemmatized) version of the corpus
        window_size: sliding window for NPMI coherence
        """
        
        coherence_model = CoherenceModel(topics = self.lemmatized_topic_words,
                                         #corpus = corpus, --> not necessary for npmi coherence
                                         dictionary = self.id2word,
                                         texts = self.texts,
                                        window_size = self.c_window_size,
                                        coherence = "c_npmi"
                                        )
        
        coherence = coherence_model.get_coherence()
        coherence_per_topic = coherence_model.get_coherence_per_topic()

        #save boxplot if plit_dir is given
        if plot_name is not None:
            plt.boxplot(coherence_per_topic)
            plt.title("Boxplot NPMI coherence per topic")
            plt.savefig(plot_name)
            plt.show()
            #plt.bar(range(len(coherence_per_topic)), coherence_per_topic)
        
            
        print(f"Coherence NPMI: {coherence:.3f}")
        print(f"Number of topics: {len(coherence_per_topic)}")
    
        print(f"Best topic: {max(coherence_per_topic):.3f}, topic: {coherence_per_topic.index(max(coherence_per_topic))}")
        print(self.lemmatized_topic_words[coherence_per_topic.index(max(coherence_per_topic))])
        
        print(f"Worst topic: {min(coherence_per_topic):.3f}, topic: {coherence_per_topic.index(min(coherence_per_topic))}")
        print(self.lemmatized_topic_words[coherence_per_topic.index(min(coherence_per_topic))])
        
        return coherence, coherence_per_topic

    def analyze_topic_quality(self):
        """
        Analyze topic quality while preserving word frequencies and properly handling bigrams.
    
        Parameters:
        - tm: BERTopic model.
        - combined_keywords: List of predefined keyword lists per topic.
        - num_words: Number of top words per topic to evaluate.
        - min_words_for_assignment: Minimum words matching a keyword list to assign a topic.
    
        Returns:
        - DataFrame with topic assignment details.
        """
        # Create keyword-to-topic mapping
        keyword_to_topic = {}
        for topic_idx, words in enumerate(self.keyword_list):
            for word in words:
                keyword_to_topic[word] = topic_idx  # Store topic index for each keyword
    
        results = []
    
        for topic_idx, words in enumerate(self.lemmatized_topic_words):
            word_counts = Counter(words)  # Preserve word frequencies
    
            # Track how many words match each predefined topic
            topic_match_counts = {i: 0 for i in range(len(self.keyword_list))}
            mixed_terms = Counter()
            TP, FP, FN = Counter(), Counter(), Counter()
    
            for word, freq in word_counts.items():
                # Handle bigrams
                if " " in word:
                    parts = word.split(" ")
                    part_topics = {keyword_to_topic[part] for part in parts if part in keyword_to_topic}
    
                    if len(part_topics) == 1:  
                        topic_match_counts[list(part_topics)[0]] += freq  #correct topic
                    elif len(part_topics) == 2:  
                        mixed_terms[word] += freq  #mixed term (from multiple topics)
                
                # Single words
                elif word in keyword_to_topic:
                    topic_match_counts[keyword_to_topic[word]] += freq
    
            # Assign topic based on most frequent matches
            assigned_topic = max(topic_match_counts, key=topic_match_counts.get)
            # Filter out the dominant topic
            filtered_topic_match_counts = {k: v for k, v in topic_match_counts.items() if k != assigned_topic}
            # Sum of occurrences of all topic words except for the dominant topic
            sum_other_topics = sum(filtered_topic_match_counts.values())
            
            if (topic_match_counts[assigned_topic] < self.min_words_for_assignment) or (sum_other_topics > self.max_other_topic_words):
                assigned_topic = None
    
            #topic is now assigned --> go through words again to correctly count TP, FP, FN
            for word, freq in word_counts.items():
                #bigram handling
                if " " in word: 
                    parts = word.split(" ")
                    part_topics = {keyword_to_topic[part] for part in parts if part in keyword_to_topic}
    
                    if len(part_topics) == 1 and assigned_topic in part_topics:
                        TP[word] += freq  #correct match
                    elif len(part_topics) == 2:
                        mixed_terms[word] += freq  #mixed terms in bigram
    
                elif word in keyword_to_topic:
                    word_topic = keyword_to_topic[word]
    
                    if word_topic == assigned_topic:
                        TP[word] += freq  #true Positive
                    else:
                        FP[word] += freq  #false Positive
    
            # False Negatives: Words expected in this topic but missing
            if assigned_topic is not None:
                for word in self.keyword_list[assigned_topic]:
                    if word not in word_counts:
                        FN[word] += 1  
    
            results.append({
                "Generated Topic": topic_idx,
                "Assigned Predefined Topic": assigned_topic if assigned_topic is not None else "Unclear",
                "True Positives (TP)": dict(TP),
                "False Positives (FP)": dict(FP),
                "False Negatives (FN)": dict(FN),
                "Mixed Terms": dict(mixed_terms),
                "TP Count": sum(TP.values()),
                "FP Count": sum(FP.values()),
                "FN Count": sum(FN.values()),
                "Mixed Count": sum(mixed_terms.values()),
            })
    
        results_df = pd.DataFrame(results)
        
        # Calculate precision (avoid division by zero)
        results_df["Precision"] = results_df["TP Count"] / (results_df["TP Count"] + results_df["FP Count"])
        results_df["Precision"].fillna(0, inplace=True)  # Replace NaN with 0 if no TP+FP exists
    
        # Print detected topics
        detected_topics = [x for x in results_df["Assigned Predefined Topic"].tolist() if isinstance(x, int)]
        print(f"Detected topics with economic foundation: {len(detected_topics)}")
        print(f"Unique detected topics: {len(np.unique(detected_topics))}/{len(self.keyword_list)}")
    
    
        # Precision of detected topics
        undetected_topics = [x for x in range(len(self.keyword_list)) if x not in detected_topics]
        
        results_df_clear_topics = results_df[results_df["Assigned Predefined Topic"] != "Unclear"]["Precision"].tolist()
        #Append number of undetected topics to penalize the model with a precision of 0
        penalty_precision = len(undetected_topics) *[0]
        results_df_clear_topics = results_df_clear_topics + penalty_precision
        print(f"Mean precision of detected topics: {np.mean(results_df_clear_topics) * 100:.2f}% (incl")
    
        print("\nUndetected topics: ")
        for topic in undetected_topics:
            print(f"Topic {topic} - {self.topic_names[topic]}")
    
        return results_df, undetected_topics, np.mean(results_df_clear_topics)

    def _plot_wordcloud_(self, plot_name=None, figsize = (20, 12)):
        """
        Plot a wordcloud based on relevant POS-tags for each topic in the model - fixed number of columns to 5
        Assign a color to each distinct topic
        """
    
        # Combine bigrams with underscores
        combined_word_list = []
        for topic in self.lemmatized_topic_words:
            combined_topic = []
            for word in topic:
                if ' ' in word:
                    combined_topic.append(word.replace(' ', '_'))
                else:
                    combined_topic.append(word)
            combined_word_list.append(combined_topic)
    
        # Calculate the number of rows and columns based on the number of topics
        num_topics = len(self.lemmatized_topic_words)
        num_cols = 5  # Fixed number of columns
        num_rows = num_topics // num_cols + (num_topics % num_cols > 0)  # Calculate the number of rows
    
        # Define a list of distinct colors --> number of topics
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
            #'#98df8a',  # light green - topic 12 and 13 not needed anymore
            #'#ff9896',  # light red
        ]
    
        # Create a mapping from keyword to topic color
        keyword_to_color = {}
        for topic_idx, keywords in enumerate(self.keyword_list):
            color = distinct_colors[topic_idx]
            for keyword in keywords:
                keyword_to_color[keyword] = color
    
        # Create subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)

        # Ensure axs is always 2D
        if num_rows == 1:
            axs = np.expand_dims(axs, axis=0)
        if num_cols == 1:
            axs = np.expand_dims(axs, axis=1)
    
        # Generate word cloud for each topic
        for i, words in enumerate(combined_word_list):
            row_index = i // num_cols
            col_index = i % num_cols
    
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
    
            wordcloud = WordCloud(width=400, height=400, background_color="white",
                                  color_func=color_func).generate_from_frequencies({word: 1 for word in words})
    
            # Plot word cloud
            axs[row_index, col_index].imshow(wordcloud, interpolation="bilinear")
            axs[row_index, col_index].set_title(f"Topic {i}", fontsize=14, fontweight='bold')
    
            # Remove axes
            axs[row_index, col_index].axis('off')
    
            # Add frame around each subplot
            for spine in axs[row_index, col_index].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
    
        # Remove empty subplots
        for i in range(num_topics, num_rows * num_cols):
            fig.delaxes(axs.flatten()[i])
    
        # Adjust spacing between subplots
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.4)
    
        # Save the figure if output_dir is specified
        if plot_name:
            directory = f"paper1/Results/BERTopic_Models/wordcloud_{plot_name}.jpg"
            plt.savefig(directory)
            print(f"Plot saved to: {directory}")
    
        plt.show()


        
    def run_coherence(self, plot_name):
        """
        Wrapper method to run coherence calculation
        """
        topic_words = self._extract_topic_words_() #extract topic words
        self.lemmatized_topic_words = [self.lemmatize_coherence_data(topic) for topic in topic_words] #lemmatize all topic representations
        self._plot_wordcloud_(plot_name) #plot the wordcloud

        results_df, undetected_topics, mean_precision = self.analyze_topic_quality()

        mean_coherence, coherence_per_topic = self.npmi_coherence(plot_name = plot_name) # calculate coherence
        precision_weighted_coherence = mean_coherence * mean_precision
        print(f"Precision-weighted coherence: {precision_weighted_coherence:.3f}")

        return {
            "mean_coherence": mean_coherence,
            "coherence_per_topic": coherence_per_topic,
            "topic_precision": mean_precision,
            "precision_weighted_coherence": precision_weighted_coherence,   
        }

    #---------------------topic similarities

    
    def _clean_embedding_matrix_(self):
        """
        Filter the embedding matrix to only contain non-noise instances (topic != -1).
        """
        #1. get topics per document and convert to numpy array
        topic_per_document = np.array(self.bertopic_model.topics_)

        #2. remove noise --> -1 topics
        mask = topic_per_document != -1
        self.embeddings_clean = self.embeddings[mask]
        self.topics_clean = topic_per_document[mask]

        print(f"Uncleaned embedding matrix shape: {self.embeddings.shape}")
        print(f"Cleaned embedding matrix shape: {self.embeddings_clean.shape}")
        print(f"Remaining documents: {len(self.topics_clean)}")
        print(f"Removed documents: {self.embeddings.shape[0] - self.embeddings_clean.shape[0]}")


    

    def _create_topic_embeddings_(self):
        """
        Create topic embeddings by taking the mean of all embeddings belonging to that specific topic.
        Returns:
            np.ndarray of shape (n_topics, embedding_dim)
        """
        unique_topics = np.unique(self.topics_clean)
        topic_embeddings = []

        #iterate over each topic
        for topic in unique_topics:
            topic_mask = self.topics_clean == topic #mask the topics
            rel_embeddings = self.embeddings_clean[topic_mask] #mask the embeddings to be only those of the specific topic
            topic_embeddings.append(np.mean(rel_embeddings, axis=0)) #take the mean of all embeddings assigned to that topic
 
        self.topic_embeddings = np.vstack(topic_embeddings) #stack topic embeddings back in a matrix
    
    def _normalize_cosine_(self, cosine_matrix):
        """
        Normalize cosine similarity values from [-1, 1] to [0, 1].
        """
        return (cosine_matrix + 1.0) / 2.0
    
    def calculate_intertopic_cosine_similarity(self, normalize:bool = True, baseline_subtract: bool = True):
        """
        Accelerated calculation of intertopic cosine similarity using cuML.
        Returns:
            cosine_sim_matrix_cpu: numpy matrix of cosine similarity between topics
            mean_upper_triangle: mean cosine similarity of the upper triangle of the similarity matrix
        """
        #1. move topic embeddings to GPU
        topic_embeddings_gpu = cp.asarray(self.topic_embeddings)

        #2. compute cosine similarity with cuml
        cosine_sim_matrix = 1 - pairwise_distances(topic_embeddings_gpu,
                                                   topic_embeddings_gpu,
                                                   metric='cosine')

        #3. bring similarit matrix back to CPU
        cosine_sim_matrix_cpu = cp.asnumpy(cosine_sim_matrix)
        assert np.all(cosine_sim_matrix_cpu >= -1) and np.all(cosine_sim_matrix_cpu <= 1) #assert similarities before normalization
        if normalize:
            cosine_sim_matrix_cpu = self._normalize_cosine_(cosine_sim_matrix_cpu) #normalize cosine similarity matrix to range from 0 to 1
            assert np.all(cosine_sim_matrix_cpu >= 0) and np.all(cosine_sim_matrix_cpu <= 1) #assert similarities after normalization

        #subtract neutral baseline (0.5) if requested
        if baseline_subtract:
            cosine_sim_matrix_cpu = cosine_sim_matrix_cpu - 0.5

        #4. extract upper triangle without diagonal
        upper_triangle_indices = np.triu_indices(cosine_sim_matrix_cpu.shape[0], k=1)
        upper_triangle_values = cosine_sim_matrix_cpu[upper_triangle_indices]

        #5. extract mean similarity
        mean_upper_triangle = np.mean(upper_triangle_values)

        print(f"Mean intertopic similarity: {mean_upper_triangle:.3f}")
        return cosine_sim_matrix_cpu, mean_upper_triangle

    def calculate_intratopic_cosine_similarity(self, normalize:bool = True, baseline_subtract:bool = True):
        """
        Calculate the mean cosine similarity between sentence embeddings of each topic
        and the corresponding topic embeddings.
        Returns:
            cosine_similarities: list of numpy arrays with cosine similarities per topic
            mean_intratopic_sim: overall mean intratopic similarity
        """
        cosine_similarities = []
        unique_topics = np.unique(self.topics_clean)

        #transform relevant variables to GPU tensors
        embeddings_gpu = cp.asarray(self.embeddings_clean)
        topics_gpu = cp.asarray(self.topics_clean)
        topic_embeddings_gpu = cp.asarray(self.topic_embeddings)

        #iterate over each unique topic
        for idx, topic in enumerate(unique_topics):
            topic_mask = (topics_gpu == topic)
            rel_embeddings = embeddings_gpu[topic_mask]

            #topic embedding at aligned index --> topic embedding for that specific topic
            topic_embedding = topic_embeddings_gpu[idx]

            #cosine similarity between the topic embedding and each embedding which is assigned to that topic
            cosine_sim_matrix = 1 - pairwise_distances(rel_embeddings,
                                                       topic_embedding.reshape(1, -1),
                                                       metric='cosine')

            sim = cp.asnumpy(cosine_sim_matrix[:, 0])

            #normalize similarity to range between 0 and 1
            if normalize:
                sim = self._normalize_cosine_(sim)

            #subtract baseline
            if baseline_subtract:
                sim = sim - 0.5  # Center at neutral baseline

            #append similarity score
            cosine_similarities.append(sim)

        #mean similarity per topic
        mean_intratopic_sim_per_topic = [np.mean(arr) for arr in cosine_similarities] #take the mean of each array to get each topic's intratopic similarity
        mean_intratopic_sim = np.mean(mean_intratopic_sim_per_topic) #take the mean of all intratopic similarities to get the overall intratopic similarity

        print(f"Mean intratopic similarity: {mean_intratopic_sim:.3f}")
        return cosine_similarities, mean_intratopic_sim

    def run_topic_similarities(self, normalize:bool = True, baseline_subtract:bool = True):
        """
        Wrapper method to perform the evaluation of topic similarities.
        Returns:
            dict with intertopic + intratopic similarity results
        """
        self._clean_embedding_matrix_() #clean the embedding matrix --> remove noise
        self._create_topic_embeddings_() #create topic embeddings

        #calculate topic similarities
        intertopic_cosine_sim_matrix, mean_upper_triangle = self.calculate_intertopic_cosine_similarity(normalize, baseline_subtract)
        intratopic_cosine_similarities, mean_intratopic_sim = self.calculate_intratopic_cosine_similarity(normalize, baseline_subtract)

        return {
            "intertopic_cosine_sim_matrix": intertopic_cosine_sim_matrix,
            "upper_triangle_intertopic_sim": mean_upper_triangle,
            "intratopic_cosine_similarities": intratopic_cosine_similarities,
            "mean_intratopic_sim": mean_intratopic_sim
        }

    def run(self, plot_name, normalize:bool = False, baseline_subtract:bool = False):
        """
        Wrapper method to run both coherence and topic similarity evaluation
        """
        #run coherence evaluation
        coherence_results = self.run_coherence(plot_name = plot_name)
    
        #extract topic precision
        topic_precision = coherence_results["topic_precision"]
    
        #run topic similarity evaluation
        topic_similarities = self.run_topic_similarities(normalize, baseline_subtract)
        if topic_precision > 0:
            weighted_intratopic_sim = topic_similarities["mean_intratopic_sim"] * topic_precision
            weighted_intertopic_sim = min(1, topic_similarities["upper_triangle_intertopic_sim"] / topic_precision) #keep intertopic_sim in range of max. 1
        else:
            #fallback when topic precision = 0
            weighted_intratopic_sim = 0.0 #penalize to 0
            weighted_intertopic_sim = float("1.0")  #penalize to 1

        print(f"Precision-weighted intratopic similarity: {weighted_intratopic_sim:.3f}")
        print(f"Precision-weighted intertopic similarity: {weighted_intertopic_sim:.3f}")
    
        topic_similarities["weighted_intratopic_sim"] = weighted_intratopic_sim
        topic_similarities["weighted_intertopic_sim"] = weighted_intertopic_sim
    
        return {
            "coherence": coherence_results,
            "topic_similarities": topic_similarities
        }