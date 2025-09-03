from cuml.neighbors import NearestNeighbors
import numpy as np
import torch
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.similarity_functions import SimilarityFunction
from sklearn.metrics import classification_report
import csv
import os
import logging
from contextlib import nullcontext

logger = logging.getLogger(__name__)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import precision_recall_fscore_support

class ClasswiseEvaluator(SentenceEvaluator):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        batch_size: int = 32,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ):
        """
        ClasswiseEvaluator.

        Args:
            texts (List[str]): List of input texts.
            labels (List[int]): Corresponding labels.
            batch_size (int): Batch size for embedding computation. Defaults to 32.
            name (str): Name for the output. Defaults to "".
            show_progress_bar (bool): Whether to display a progress bar during embedding computation.
                Defaults to False.
            write_csv (bool): Whether to log results to a CSV file. Defaults to True.
            truncate_dim (int, optional): Dimension to truncate embeddings to. Defaults to None.
        """
        super().__init__()
        self.texts = texts
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.name = name
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv
        self.truncate_dim = truncate_dim
        self.csv_file = f"classwise_evaluation_{name}_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", "precision_macro", "recall_macro",
                            "f1_weighted", "f1_macro", "roc_auc"]

    @classmethod
    def from_input_examples(cls, examples, **kwargs):
        texts = [example.texts[0] for example in examples]
        labels = [example.label for example in examples]
        return cls(texts, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1):
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"ClasswiseEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            embeddings = model.encode(
                self.texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )

        # Compute class centroids
        unique_labels = np.unique(self.labels)
        centroids = {label: np.mean(embeddings[self.labels == label], axis=0) for label in unique_labels}

        # Nearest centroid predictions 
        predictions = self._nearest_centroid_predict(embeddings, centroids)

        # Metrics computation
        metrics = self._compute_metrics(predictions)

        logger.info(f"Metrics: {metrics}")

        # Display Confusion Matrix in Terminal
        self._display_confusion_matrix(predictions)

        #display the class scores
        self._print_class_scores(predictions)

        # Write results to CSV
        if output_path and self.write_csv:
            self._write_to_csv(output_path, metrics, epoch, steps)

        self.store_metrics_in_model_card_data(model, metrics)
        return metrics

    def _nearest_centroid_predict(self, embeddings, centroids):
        """
        predict class for datapoint based on the nearest topic centroid
        """
        centroid_matrix = np.array(list(centroids.values()))
        labels = list(centroids.keys())

        distances = np.linalg.norm(embeddings[:, None] - centroid_matrix[None, :], axis=2)
        nearest_centroid_indices = np.argmin(distances, axis=1)
        predictions = np.array([labels[i] for i in nearest_centroid_indices])

        return predictions

    def _compute_metrics(self, predictions):
        """
        compute the classification evaluation metrics accuracym precision, recall, f1-score and roc-auc
        """
        accuracy = accuracy_score(self.labels, predictions)
        precision_macro = precision_score(self.labels, predictions, average="macro", zero_division=0)
        recall_macro = recall_score(self.labels, predictions, average="macro", zero_division=0)
        f1_macro = f1_score(self.labels, predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(self.labels, predictions, average="weighted", zero_division=0)

        # Compute ROC-AUC (requires one-hot encoded labels and probabilities)
        try:
            roc_auc = roc_auc_score(
                np.eye(len(np.unique(self.labels)))[self.labels],
                np.eye(len(np.unique(self.labels)))[predictions],
                multi_class="ovr"
            )
        except ValueError:  # Handle cases with a single class in predictions
            roc_auc = float("nan")

        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "roc_auc": roc_auc,
        }

    def _display_confusion_matrix(self, predictions):
        """
        Display the confusion matrix as text in the terminal.
        """
        matrix = confusion_matrix(self.labels, predictions)
        unique_labels = np.unique(self.labels)
        headers = [""] + [str(label) for label in unique_labels]
        table = [[str(unique_labels[i])] + row.tolist() for i, row in enumerate(matrix)]
        print("\nConfusion Matrix:")
        print(tabulate(table, headers, tablefmt="grid"))

    def _print_class_scores(self, predictions):
        """
        Print precision, recall, F1-score, and support for each class.
        """
        unique_labels = np.unique(self.labels)
        precision, recall, f1, support = precision_recall_fscore_support(self.labels, predictions, zero_division=0)
        
        # Prepare a table for display
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        table = [
            [str(unique_labels[i]), f"{precision[i]:.2f}", f"{recall[i]:.2f}", f"{f1[i]:.2f}", support[i]]
            for i in range(len(unique_labels))
        ]
        print("\nClass-wise Metrics:")
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def _write_to_csv(self, output_path, metrics, epoch, steps):
        csv_path = os.path.join(output_path, self.csv_file)
        if not os.path.isfile(csv_path):
            with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
                writer.writerow([epoch, steps] + list(metrics.values()))
        else:
            with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, steps] + list(metrics.values()))
