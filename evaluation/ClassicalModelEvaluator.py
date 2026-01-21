import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import cupy as cp
from cuml.metrics import pairwise_distances
from collections import Counter
from gensim.models import CoherenceModel
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm



class ClassicalModelEvaluator:
    """
    Class to evaluate the performance of classical topic models
    """
    def __init__(
        self,
        model,  #topic model
        texts,  #texts/tokens for coherence calculation
        id2word,  #gensim dictionary
        corpus, #gensim corpus
        keywords, #list of lists of keywords
        topic_names, #list of topic names
        n_topic_words: int = 5,  #number of topic words to consider
        min_words_for_assignment:int=2, #minimum number of words from one topic which need to be present in a topic to assign it as "dominant" topic
        max_other_topic_words:int = 1, #maximum number of words from other topics without discarding that topic as not defined
        c_window_size:int = 10
    ):
    
        self.model = model
        self.texts = texts
        self.id2word = id2word
        self.corpus = corpus
        self.keywords = keywords
        self.topic_names = topic_names
        self.n_topic_words = n_topic_words
        self.min_words_for_assignment = min_words_for_assignment
        self.max_other_topic_words = max_other_topic_words
        self.c_window_size = c_window_size

        self.topic_words = None
        self.betas = None

    def _extract_topic_words_(self):
        """
        Extract the top-n words per topic from the model
        """
        topics = self.model.show_topics(
            num_topics = -1, #ensure that all topics are extracted
            num_words = self.n_topic_words, #set number of top-words to extract
            formatted = False
        )
        
        #extract words and betas separately
        self.topic_words = [[word for word, beta in word_prob_list] for _, word_prob_list in topics]
        self.betas = [[beta for word, beta in word_prob_list] for _, word_prob_list in topics]



    def plot_wordcloud(self, plot_name = None):
        """
        Plot a wordcloud based on relevant POS-tags for each topic in the model - fixed number of columns to 5
        Assign a color to each distinct topic
        """
        self._extract_topic_words_()
    
        # Calculate the number of rows and columns based on the number of topics
        num_topics = len(self.keywords)
        num_cols = 5  # Fixed number of columns
        num_rows = num_topics // num_cols + (num_topics % num_cols > 0)  # Calculate the number of rows
    
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
        #'#98df8a',  # light green topic 12 and 13 not needed anymore
        #'#ff9896',  # light red
        ]
        
        # Create a mapping from keyword to topic color
        keyword_to_color = {}
        for topic_idx, words in enumerate(self.keywords):
            color = distinct_colors[topic_idx]
            for word in words:
                keyword_to_color[word] = color
    
        # Create subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))
    
        # Generate word cloud for each topic
        for i, (words, probs) in enumerate(zip(self.topic_words, self.betas)):
            row_index = i // num_cols
            col_index = i % num_cols
    
            # Create a color function that uses unique colors for topic keywords and black for others
            def color_func(word, *args, **kwargs):
                return keyword_to_color.get(word, "black")
    
            wordcloud = WordCloud(width=200, height=200, background_color="white",
                                  color_func=color_func).generate_from_frequencies(dict(zip(words, probs)))
    
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
    
        plt.tight_layout()
    
        # Save the figure if output_dir is specified
        if plot_name is not None:
            directory = f"paper1/Results/Classical_Models/wordcloud_{plot_name}.jpg"
            plt.savefig(directory)
            print(f"Plot saved to: {directory}")
        
        plt.show()


    def analyze_topic_quality(self):
        """
        Analyze topic quality while preserving word frequencies and properly handling bigrams.
    
        Parameters:
        - tm: gensim model.
        - corpus: gensim corpus.
        - combined_keywords: List of predefined keyword lists per topic.
        - num_words: Number of top words per topic to evaluate.
        - min_words_for_assignment: Minimum words matching a keyword list to assign a topic.
    
        Returns:
        - DataFrame with topic assignment details.
        """
        
        # Create keyword-to-topic mapping
        keyword_to_topic = {}
        for topic_idx, words in enumerate(self.keywords):
            for word in words:
                keyword_to_topic[word] = topic_idx  # Store topic index for each keyword
    
        results = []
    
        for topic_idx, words in enumerate(self.topic_words):
            word_counts = Counter(words)  # Preserve word frequencies
    
            # Track how many words match each predefined topic
            topic_match_counts = {i: 0 for i in range(len(self.keywords))}
            mixed_terms = Counter()
            TP, FP, FN = Counter(), Counter(), Counter()
    
            for word, freq in word_counts.items():
                # Handle bigrams
                if "_" in word:
                    parts = word.split("_")
                    part_topics = {keyword_to_topic[part] for part in parts if part in keyword_to_topic}
    
                    if len(part_topics) == 1:  
                        topic_match_counts[list(part_topics)[0]] += freq  # Correct topic
                    elif len(part_topics) == 2:  
                        mixed_terms[word] += freq  # Mixed term (from multiple topics)
                
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
    
            # Now that the topic is assigned, go through words again to correctly count TP, FP, FN
            for word, freq in word_counts.items():
                if " " in word:  # Bigram handling
                    parts = word.split(" ")
                    part_topics = {keyword_to_topic[part] for part in parts if part in keyword_to_topic}
    
                    if len(part_topics) == 1 and assigned_topic in part_topics:
                        TP[word] += freq  # Correct match
                    elif len(part_topics) == 2:
                        mixed_terms[word] += freq  # Mixed term
    
                elif word in keyword_to_topic:
                    word_topic = keyword_to_topic[word]
    
                    if word_topic == assigned_topic:
                        TP[word] += freq  # True Positive
                    else:
                        FP[word] += freq  # False Positive
    
            # False Negatives: Words expected in this topic but missing
            if assigned_topic is not None:
                for word in self.keywords[assigned_topic]:
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
        print(f"Unique detected topics: {len(np.unique(detected_topics))}/{len(self.keywords)}")
    
        
    
        # Precision of detected topics
        undetected_topics = [x for x in range(len(self.keywords)) if x not in detected_topics]
        
        results_df_clear_topics = results_df[results_df["Assigned Predefined Topic"] != "Unclear"]["Precision"].tolist()
        #Append number of undetected topics to penalize the model with a precision of 0
        penalty_precision = len(undetected_topics) *[0]
        results_df_clear_topics = results_df_clear_topics + penalty_precision
        print(f"Mean precision of detected topics: {np.mean(results_df_clear_topics) * 100:.2f}% (incl")
    
        print("\nUndetected topics: ")
        for topic in undetected_topics:
            print(f"Topic {topic} - {self.topic_names[topic]}")
    
        return results_df, undetected_topics, np.mean(results_df_clear_topics)



    def npmi_coherence(self):
        """
        Function to calculate coherence of the classical models
        1. Extract relevant keywords from the model (based on POS-tags to have more meaningful topics)
        2. Create a gensim coherence model
        3. Calculate coherence
    
        input:
        model: gensim topic model
        corpus: gensim corpus of the model used as input
        dictionary: gensim id2word of the model used as input
        texts: needed for sliding window approaches like npmi coherence --> original, cleaned texts/tokens
        window_size: integer, size of the sliding window
        """
    
    
        # Check if any topic or text is empty and filter out empty topics
        filtered_topics = []
        for i, topic in enumerate(self.topic_words):
            if topic:
                filtered_topics.append(topic)
            else:
                print(f"Warning: Topic {i} is empty!")
    
        if not filtered_topics:
            print("No valid topics found. Unable to calculate coherence.")
            return None, None
        
        empty_text_idx = []
        for i, text in enumerate(self.texts):
            if not text:
                empty_text_idx.append(i)
        print(f"Warning: {len(empty_text_idx)} empty documents")
    
        print(f"Number of empty topics: {len(self.keywords) - len(filtered_topics)}")
                
        #2. Create coherence model
        coherence_model = CoherenceModel(
            topics = filtered_topics,
            #corpus = corpus, --> corpus not necessary in this context
            dictionary = self.id2word,
            texts = self.texts,
            window_size = self.c_window_size,
            coherence = "c_npmi")
    
        #3. Calculate coherence and plot the results
        coherence = coherence_model.get_coherence()
        coherence_per_topic = coherence_model.get_coherence_per_topic()
    
    
        valid_coherence_scores = [score for score in coherence_per_topic if np.isfinite(score)]
    
        if not valid_coherence_scores:
            print("No valid coherence scores calculated. All scores are infinite.")
            return None
    
        average_coherence = np.mean(valid_coherence_scores)
    
        print(f"Number of infinite coherence values: {len(coherence_per_topic) - len(valid_coherence_scores)}")
    
        print (f"Coherence NPMI: {average_coherence:.3f}")
        print(f"Coherence NPMI per topic: {valid_coherence_scores}")
        print(f"Best topic: {max(valid_coherence_scores):.3f}, topic: {valid_coherence_scores.index(max(valid_coherence_scores))}")
        #print(topics[valid_coherence_scores.index(max(valid_coherence_scores))])
        
        print(f"Worst topic: {min(valid_coherence_scores):.3f}, topic: {valid_coherence_scores.index(min(valid_coherence_scores))}")
        #print(topics[valid_coherence_scores.index(min(valid_coherence_scores))])
    
        plt.boxplot(valid_coherence_scores)
        plt.title("Boxplot NPMI coherence per topic")
        plt.show()
        
        return average_coherence, valid_coherence_scores

    def run_coherence(self, plot_name = None):
        #plot the wordcloud
        self.plot_wordcloud(plot_name)
        results_df, undetected_topics, mean_precision = self.analyze_topic_quality()
        avg_coherence, coherence_per_topic = self.npmi_coherence()

        precision_weighted_coherence = avg_coherence * mean_precision
        print(f"Precision-weighted coherence: {precision_weighted_coherence:.3f}")

        return {
            "mean_coherence": avg_coherence,
            "coherence_per_topic": coherence_per_topic,
            "topic_precision": mean_precision,
            "precision_weighted_coherence": precision_weighted_coherence,   
        }
        


    #-----------------Topic similarities

    def _check_rowsums_1_(self, matrix, tolerance = 1e-5):
        """
        Check if the rowsums of a matrix are equal to 1
        """
        rowsums = matrix.sum(axis=1)
        all_good = np.allclose(rowsums, 1, atol=tolerance)
        if all_good:
            return "All rows sum to 1"
        else:
            bad_rows = np.where(~np.isclose(rowsums, 1, atol=tolerance))[0]
            return f"Rows {bad_rows} do not sum to 1"

    def _get_beta_matrix_(self, sparsity_threshold = 0.0001):
        """
        obtain the beta matrix of a topic model
    
        Input:
        """
        beta_matrix = self.model.get_topics()
    
        non_zero_elements = np.count_nonzero(np.abs(beta_matrix) >= sparsity_threshold)  # Count values above the threshold
        total_elements = beta_matrix.size  # Total number of elements in the matrix
    
        # Calculate sparsity
        sparsity = 1 - (non_zero_elements / total_elements)
    
        print(f"Sparsity of the beta matrix with threshold {sparsity_threshold}: {sparsity:.4f}")
        print(f"Shape of beta matrix: {beta_matrix.shape}")
        
        return beta_matrix
    
    def _get_gamma_matrix_(self, sparsity_threshold = 0.0001):
        """
        extract the gamma matrix from a classical topic model

        """
        num_topics = self.model.num_topics
        
        #create list to store results
        result_gamma = []
    
        #iterate over each document and obtain the gamma values
        for doc in tqdm(range(0, len(self.corpus)), desc = "Extract gamma values"):
            gamma_per_doc = self.model.get_document_topics(self.corpus[doc], minimum_probability = 0) #get gamma values for each document in corpus, set min prob to 0 so that each gamma is extracted
            topic = [element[0] for element in gamma_per_doc]  #extract topic
            gamma_value = [element[1] for element in gamma_per_doc] #extract gamma value
            
            result_gamma.append((doc, topic, gamma_value)) #append results to list
    
        doc_indices = set(item[0] for item in result_gamma)
        
        #create gamma matrix
        gamma_matrix = np.zeros((len(doc_indices), num_topics))
    
        #fill matrix with gamma values by iterating over the results per document
        for item in tqdm(result_gamma, desc = "Fill gamma matrix"):
            doc_index = item[0]
            topic_index = item[1]
            gamma_values = item[2]
    
            gamma_matrix[doc_index, topic_index] = gamma_values
    
        
        #ensure that each row of the gamma matrix sums to 1
        self._check_rowsums_1_(gamma_matrix)
        non_zero_elements = np.count_nonzero(np.abs(gamma_matrix) >= sparsity_threshold)  # Count values above the threshold
        total_elements = gamma_matrix.size  # Total number of elements in the matrix
    
        # Calculate sparsity
        sparsity = 1 - (non_zero_elements / total_elements)
    
        print(f"Sparsity of the gamma matrix with threshold {sparsity_threshold}: {sparsity:.4f}")
        print(f"Shape of gamma matrix: {gamma_matrix.shape}")
        
        return gamma_matrix
        
    
    
    def _determine_main_topic_(self, gamma_matrix):
        """
        determine the main topic of each document in the gamma matrix by obtaining the maximum gamma value per document
    
        Input:
        gamma_matrix: gamma matrix
        """
        result_mapped_topics = []
        for i in tqdm(range(len(gamma_matrix)), desc = "Determine main topic"): #iterate through each document 
            dominant_topic = np.argmax(gamma_matrix[i,:])  #determine dominant topic for each document based on highest gamma value per documents
            dominant_topic_value = gamma_matrix[i, dominant_topic]  # Get the gamma value of the dominant topic
            result_mapped_topics.append([dominant_topic, dominant_topic_value])  #append result to list
    
        main_topics = [topic for topic, gamma in result_mapped_topics]
        plt.hist(main_topics, edgecolor = "white")
        plt.title("Histogram of number of documents per topic")
        plt.show()
    
            
        return result_mapped_topics
    
    def get_beta_gamma_matrix(self):
        """
        Wrapper method to get beta and gamma matrix as well as the main topic per document
        """
        beta_matrix = self._get_beta_matrix_()
        gamma_matrix = self._get_gamma_matrix_()
        main_topics = self._determine_main_topic_(gamma_matrix)
            
        return beta_matrix, gamma_matrix, main_topics

    def _baseline_adjust_(self, sim_array, baseline: float = 0.5):
        """
        Shift similarity scores by a neutral baseline.
        sim_array: np.array or list of similarities in [0,1]
        baseline: neutral similarity
        Returns array centered at zero
        """
        return np.array(sim_array) - baseline
    
        
    def intertopic_similarity_gamma(self, topic_embeddings):
        """
        Calculate intertopic similarity using the gamma embeddings with cuML.
        """
        try:
            # Stack embeddings and move to GPU
            topic_embeddings_matrix = cp.asarray(np.vstack([np.asarray(te).flatten() for te in topic_embeddings]))
    
            # Compute cosine distances and convert to similarity
            dist_matrix = pairwise_distances(topic_embeddings_matrix, metric="cosine")
            similarity_matrix = 1 - dist_matrix.get()  # move back to CPU
    
            # Only upper triangle
            upper_triangle_indices = np.triu_indices(similarity_matrix.shape[0], k=1)
            upper_triangle_values = similarity_matrix[upper_triangle_indices]
    
            plt.boxplot(upper_triangle_values)
            plt.title("Boxplot Intertopic Similarities (gamma)")
            plt.show()
    
            intertopic_sim = np.mean(upper_triangle_values)
            print(f"Intertopic Similarity (gamma): {intertopic_sim:.3f}")
            print(f"Standard deviation (gamma): {np.std(upper_triangle_values):.3f}")
    
        except Exception as e:
            print(f"Error computing gamma similarity: {e}")
            intertopic_sim = 1  # Penalize if computation fails
            print("Intertopic similarity (gamma) penalized to 1")
    
        return intertopic_sim
    
    
    def intertopic_similarity_beta(self, beta_matrix):
        """
        Calculate intertopic similarity using the beta matrix with cuML.
        """
        beta_gpu = cp.asarray(beta_matrix)
        dist_matrix = pairwise_distances(beta_gpu, metric="cosine")
        similarity_matrix = 1 - dist_matrix.get()  # convert to similarity on CPU
    
        # Only upper triangle values
        upper_triangle_indices = np.triu_indices(similarity_matrix.shape[0], k=1)
        upper_triangle_values = similarity_matrix[upper_triangle_indices]
    
        plt.boxplot(upper_triangle_values)
        plt.title("Boxplot Intertopic Similarities (beta)")
        plt.show()
    
        intertopic_sim = np.mean(upper_triangle_values)
        print(f"Intertopic Similarity (beta): {intertopic_sim:.3f}")
        print(f"Standard deviation (beta): {np.std(upper_triangle_values):.3f}")
    
        return intertopic_sim, similarity_matrix, upper_triangle_values

    
    def intratopic_similarity(self, gamma_matrix, main_topics):
        """
        Calculate the intratopic similarity of classical topic models using cuML.
    
        Each document has a dominant topic (highest gamma value). 
        The intratopic similarity measures the cohesion of documents within a topic.
    
        Returns:
            mean_intratopic_similarity: List of mean similarities per topic
            overall_mean_similarity: Overall mean similarity across all topics
            topic_embeddings: List of topic embeddings
            all_similarities: List of similarity arrays per topic
            topic_int: list of dominant topic indices per document
        """
    
        topic_int = [topic for topic, gamma in main_topics]
        num_topics = gamma_matrix.shape[1]
        unique_topics = np.unique(topic_int)
    
        # Penalize if some topics have no documents
        if len(unique_topics) < num_topics:
            print(f"Number of unique topics: {len(unique_topics)}")
            print("Intratopic similarity penalized to 0")
            return [], 0.0, [], [], topic_int
    
        # Prepare storage
        mean_intratopic_similarity = []
        topic_embeddings = []
        all_similarities = []
    
    
        for topic_id in unique_topics:
            # Select documents assigned to this topic
            doc_indices = [i for i, t in enumerate(topic_int) if t == topic_id]
            docs = gamma_matrix[doc_indices]
    
            if len(docs) > 1:
                # Topic embedding = mean of assigned documents
                topic_embedding = np.mean(docs, axis=0, keepdims=True)
                topic_embeddings.append(topic_embedding)
    
                # Compute cosine similarity of docs to topic embedding
                similarities = cosine_similarity(topic_embedding, docs)
                all_similarities.append(similarities)
    
                mean_similarity = np.mean(similarities)
                mean_intratopic_similarity.append(mean_similarity)
    
        overall_mean_similarity = np.mean(mean_intratopic_similarity)
        print(f"Intratopic similarity: {overall_mean_similarity:.3f}")
    
        return mean_intratopic_similarity, overall_mean_similarity, topic_embeddings, all_similarities, topic_int
    
    
    def run_topic_similarities(self):
        """
        Wrapper method to calculate intratopic and intertopic similarities
        """
        beta_matrix, gamma_matrix, main_topics = self.get_beta_gamma_matrix()
    
        intratopic_sim, overall_mean_sim, topic_embeddings, all_sims, topic_int = self.intratopic_similarity(gamma_matrix, main_topics)
        intertopic_sim_beta, _, _ = self.intertopic_similarity_beta(beta_matrix)
        intertopic_sim_gamma = self.intertopic_similarity_gamma(topic_embeddings)
    
        return {
            "intratopic_similarity": intratopic_sim,
            "overall_intratopic_similarity": overall_mean_sim,
            "intertopic_similarity_beta": intertopic_sim_beta,
            "intertopic_similarity_gamma": intertopic_sim_gamma,
            "main_topics": main_topics
        }

    def run(self, plot_name):
        """
        Wrapper method to run both coherence and topic similarity evaluation
        """
        #run coherence evaluation
        coherence_results = self.run_coherence(plot_name = plot_name)

        #extract topic precision
        topic_precision = coherence_results["topic_precision"]

        #run topic similarity evaluation
        topic_similarities = self.run_topic_similarities()
        if topic_precision > 0:
            weighted_intratopic_sim = topic_similarities["overall_intratopic_similarity"] * topic_precision
            weighted_intertopic_sim_gamma = min(1, topic_similarities["intertopic_similarity_gamma"] / topic_precision) #keep intertopic_sim in range of max. 1
        else:
            #fallback when topic precision = 0
            weighted_intratopic_sim = 0.0 #penalize to 0
            weighted_intertopic_sim_gamma = float("1.0")  #penalize to 1

        print(f"Precision-weighted intratopic similarity: {weighted_intratopic_sim:.3f}")
        print(f"Precision-weighted intertopic similarity (gamma): {weighted_intertopic_sim_gamma:.3f}")

        topic_similarities["weighted_intratopic_sim"] = weighted_intratopic_sim
        topic_similarities["weighted_intertopic_sim_gamma"] = weighted_intertopic_sim_gamma

        return {
            "coherence": coherence_results,
            "topic_similarities": topic_similarities
        }

        
