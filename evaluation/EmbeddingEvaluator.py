import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import cupy as cp
from cuml.metrics import pairwise_distances
import umap
import plotly.express as px
from sklearn.utils import shuffle
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support
)
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

class EmbeddingEvaluator:
    """
    Evaluate embeddings based on the testset (intra- and intertopic similarity) and on the number of outliers within the BERTopic framework
    """
    def __init__(self, embeddings, topics, sentences, topic_names, bertopic_model = None):
        self.embeddings = embeddings
        self.topics = topics
        self.sentences = sentences
        self.topic_names = topic_names
        self.bertopic_model = bertopic_model

        #create topic embeddings
        self.topic_embeddings = self.create_topic_embeddings()

    
    def create_topic_embeddings(self):
        """
        Function to create topic embeddings by taking the mean of all embeddings belonging to that specific topic
    
        Returns a matrix with embeddings per topic: dimensions: num_topics x model dimensions
        """
        #get unique topics
        unique_topics = np.unique(self.topics)
        topic_embeddings = []
       
        #iterate over each topic
        for topic in unique_topics:
            #extract embeddings for the current topic
            topic_mask = self.topics == topic
            rel_embeddings = self.embeddings[topic_mask]
            topic_embeddings.append(np.mean(rel_embeddings, axis=0)) #take the mean of the embeddings for that topic to get topic embeddings
    
        return np.array(topic_embeddings)
    
    
    def calculate_intertopic_cosine_similarity(self):
        """
        Accelerated calculation of intertopic cosine similarity using cuML.
    
        Args:
            embeddings: original sentence embedding matrix.
            tm: BERTopic model.
    
        Returns:
            row_similarities: List of pairwise cosine similarities between topic embeddings.
        """
        
        # 4. Move topic embeddings to the GPU
        topic_embeddings_gpu = cp.asarray(self.topic_embeddings)  # Move to GPU
    
        # 5. Compute the pairwise cosine similarities using cuML's GPU-accelerated function
        cosine_sim_matrix = 1 - pairwise_distances(topic_embeddings_gpu, topic_embeddings_gpu, metric='cosine')
    
        # 6. Convert the matrix to CPU for further processing (optional)
        cosine_sim_matrix_cpu = cp.asnumpy(cosine_sim_matrix)
    
    
        # Extract upper triangle without diagonal
        upper_triangle_values = cosine_sim_matrix_cpu[np.triu_indices_from(cosine_sim_matrix_cpu, k=1)]
    
        # Compute mean similarity
        mean_upper_triangle = np.mean(upper_triangle_values)
    
        print(f"Mean intertopic similarity: {mean_upper_triangle:.3f}")
    
        # 8. Return the row-wise similarities
        return cosine_sim_matrix_cpu, mean_upper_triangle

    def __plot_intertopic_sim_matrix__(self, cosine_sim_matrix, fig_name = None):
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cosine_sim_matrix, annot=True, cmap="coolwarm", center=0, cbar_kws={'label': 'Cosine Similarity'},
            xticklabels=self.topic_names, yticklabels=self.topic_names
        )
        plt.xlabel("Topic")
        plt.ylabel("Topic")
        plt.title("Cosine Similarity Between Topics")
        #plt.xticks(self.topic_names)
        
        if fig_name is not None:
            file_path = f"paper1/Results/Embeddings/test_data/cosine_sim_matrix_{fig_name}.jpg"
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved as {file_path}")
        plt.show()
        plt.close()
    
    
    def calculate_intratopic_cosine_similarity(self):
        """
        Calculate the mean cosine similarity between sentence embeddings of each topic
        and the corresponding topic embeddings.
    
        Args:
            embeddings: numpy array of shape (num_sentences, embedding_dim) containing sentence embeddings.
            topics: list of topic labels for each sentence.
            topic_embeddings: numpy array of shape (num_topics, embedding_dim) containing topic embeddings.
    
        Returns:
            mean_cosine_similarities: list with num_topics elements, containing the mean cosine similarities
                                       between sentence embeddings of each topic and the corresponding topic embeddings.
        """
        # Move embeddings and topics to GPU
        
        cosine_similarities = []
    
        unique_topics = np.unique(self.topics)
        print(f"Number of topics: {len(unique_topics)}")
    
        #transform np into cupy for GPU acceleration
        embeddings_gpu = cp.asarray(self.embeddings)
        topics_gpu = cp.asarray(self.topics)
        topic_embeddings_gpu = cp.asarray(self.topic_embeddings)
        
        for topic in unique_topics:
            # Extract embeddings for the current topic
            topic_mask = self.topics == topic
    
            #get the embeddings related to the topic
            rel_embeddings = embeddings_gpu[topic_mask]
            
            #get the corresponding topic embedding
            topic_embedding = topic_embeddings_gpu[topic]
    
            #create list to store topic_similartiy_scores
            topic_similarity_scores = []
    
            # Calculate cosine similarity between each sentence embedding and topic embedding
            cosine_sim_matrix = 1 - pairwise_distances(rel_embeddings, topic_embedding.reshape(1, -1), metric='cosine')
    
            # Store the cosine similarity scores
            cosine_similarities.append(cosine_sim_matrix[:, 0].tolist())  # Convert from CuPy array to list

        print(f"Number of topics: {len(cosine_similarities)}")
        mean_intratopic_sim_per_topic = [np.mean(lst) for lst in cosine_similarities]
        mean_intratopic_sim = np.mean(mean_intratopic_sim_per_topic)
        print(f"Mean intratopic similarity: {mean_intratopic_sim:.3f}")
    
            
        return cosine_similarities, mean_intratopic_sim
    

    def get_bertopic_outliers(self):
        """
        Extract the number of outliers from BERTopic model
        """
        topics_tm = self.bertopic_model.topics_
        num_outliers = topics_tm.count(-1)
        print(f"Number of outliers: {num_outliers}")
        
        return num_outliers
        
    def run(self, fig_name):
        """
        wrapper method to evaluate quality of embeddings --> intra- and intertopic similarity (raw embeddings on testset) and number of outliers in combination with BERTopic
        """
        intratopic_sims, mean_intratopic_sim = self.calculate_intratopic_cosine_similarity()
        intertopic_sim_matrix, mean_intertopic_sim = self.calculate_intertopic_cosine_similarity()

        self.__plot_intertopic_sim_matrix__(intertopic_sim_matrix, fig_name = fig_name)

        if self.bertopic_model is not None:
            num_outliers = self.get_bertopic_outliers()
        else:
            num_outliers = "No BERTopic model provided"

        results = {
            "intratopic_similarities": intratopic_sims,
            "mean_intratopic_similarity": mean_intratopic_sim,
            "intertopic_similarity_matrix": intertopic_sim_matrix,
            "mean_intertopic_similarity": mean_intertopic_sim,
            "num_outliers": num_outliers
        }
        return results
        
    def __reduce_dimensionality__(self, n_neighbors:int = 125, n_components:int = 2, min_dist:float = 0.0, metric:str = "cosine"):
        """
        Reduce dimensionality of the embeddings. Use UMAP package as cuml leads to errors
        """
        # Start UMAP without PCA but with unique = True
        umap_model = umap.UMAP(
            n_neighbors = n_neighbors,
            n_components = n_components,
            min_dist = min_dist,
            unique = True, #very important: duplicates can create significant issues with UMAP, especially when more duplicates than nearest neighbors (https://github.com/lmcinnes/umap/issues/771#issuecomment-931886015)
            metric = metric
        )
        
        red_embeddings = umap_model.fit_transform(self.embeddings)
        self.red_embeddings = red_embeddings
        return red_embeddings



    def __plot_embeddings__(self, opacity=0.75, title=None, add_annotations=False, fig_name=None, shuffle_data = True):
        """
        Plot the embeddings with colors assigned based on the labels.
        """
        n_dim = self.red_embeddings.shape[1]  # Number of remaining dimensions after reducing
      
        # Create a DataFrame with embeddings and cluster labels
        df = pd.DataFrame(self.red_embeddings, columns=['Dimension 1', 'Dimension 2'])
        df['sentence'] = self.sentences
        df['topic'] = self.topics
        df['size'] = 2
    
        # Shuffle the data if requested
        if shuffle_data:
            df = shuffle(df).reset_index(drop=True)
        
        # Plot interactive scatter plot using Plotly Express
        fig = px.scatter(df, x='Dimension 1', y='Dimension 2', color='topic', 
                         title=title,
                         hover_name='sentence',
                         size='size',
                         opacity=opacity)
    
        if add_annotations:
            # Add annotations for labels
            for i, topic in enumerate(self.topics):
                fig.add_annotation(x=self.red_embeddings[i, 0], y=self.red_embeddings[i, 1], text=sentence, showarrow=False)
    
    
        # Remove the colorbar
        fig.update_traces(marker=dict(showscale=False))
        # Remove the colorbar
        fig.update_layout(coloraxis_showscale=False)
    
        # Update layout to adjust the size of the plot, remove title and axis titles
        fig.update_layout(
            width=1080, 
            height=720,
            title_text='',  # Remove title
            xaxis_title='',  # Remove x-axis title (2D)
            yaxis_title='',  # Remove y-axis title (2D)
            scene=dict(  # For 3D plots
                xaxis_title='',
                yaxis_title='',
                zaxis_title=''
            ) if n_dim == 3 else {},
            margin=dict(l=0, r=7, t=7, b=0)  # Reduce margins
        )
    
        if fig_name is not None:
            file_path = f"paper1/Results/Embeddings/test_data/red_embeddings_{fig_name}.jpg"
            fig.write_image(file_path, scale=2)  # scale=2 → higher resolution
            print(f"Plot saved as {file_path}")
        
        # Show plot
        fig.show()
        return None #use it to avoid plotly to print the full JSON/HTML representation
        
    def plot(self, n_neighbors:int = 100, n_components:int = 2, min_dist:float = 0.0, metric:str = "cosine", fig_name = None):
        """
        Wrapper function to reduce dimensionality and plot + optionally save the plot of reduced embeddings
        """
        #reduce dimensionality
        print("Reduce dimensionality")
        self.__reduce_dimensionality__(
            n_neighbors = n_neighbors,
            n_components = n_components,
            min_dist = min_dist,
            metric = metric,
        )

        #plot the results with optional saving
        self.__plot_embeddings__(fig_name = fig_name)
