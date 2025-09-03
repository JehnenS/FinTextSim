import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from collections.abc import Iterable

from torch import Tensor

class CircleLossText(nn.Module):
    def __init__(self, 
                 model, 
                 scale: float = 32, 
                 margin: float = 0.25, 
                 max_scale: float = 60, 
                 min_margin: float = 0.1, 
                 max_forward_passes: int = 400, 
                 debug=False):
        """
        Circle Loss for text embeddings with topic-based positive/negative pairs.

        Args:
            model: SentenceTransformer model used for generating embeddings.
            scale: Initial scale factor for the exponential terms.
            margin: Initial margin for determining the weight of positive/negative pairs.
            max_scale: Maximum value for scale to avoid it growing too large.
            min_margin: Minimum value for margin to avoid it shrinking too much.
            max_forward_passes: Total number of forward passes (used for adapting scale/margin).
            debug: Whether to print debugging information.
        """
        super().__init__()
        self.sentence_embedder = model
        self.base_scale = scale
        self.base_margin = margin
        self.max_scale = max_scale
        self.min_margin = min_margin
        self.max_forward_passes = max_forward_passes
        self.forward_passes = 0  # Counter for the number of forward passes
        self.debug = debug

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor):
        """
        Forward pass for Circle Loss.

        Args:
            sentence_features: A batch of input features for the SentenceTransformer model.
            labels: Tensor of shape (batch_size,) - topic labels for the embeddings.

        Returns:
            loss: Circle loss value for the batch.
        """
        embeddings = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        
        # Adapt scale and margin based on the number of forward passes
        self.forward_passes += 1
        #self.scale = self.base_scale * (1 + self.forward_passes / self.max_forward_passes)
        self.scale = self.base_scale + ((self.forward_passes / self.max_forward_passes) * (self.max_scale - self.base_scale))
        self.scale = min(self.scale, self.max_scale)  # Ensure the scale doesn't exceed max_scale
        
        self.margin = self.base_margin * (1 - min(self.forward_passes / self.max_forward_passes, 1))
        self.margin = max(self.margin, self.min_margin)  # Ensure the margin doesn't go below min_margin

        loss = self.circle_loss(labels, embeddings)

        if self.forward_passes == self.max_forward_passes:
            print("Final stage reached")
            print(f"Margin: {self.margin}")
            print(f"Scale: {self.scale}")

        return loss

    def circle_loss(self, labels: Tensor, embeddings: Tensor) -> Tensor:
        """
        Circle loss calculation based on positive/negative pair similarities.

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim) - sentence embeddings.
            labels: Tensor of shape (batch_size,) - topic labels for the embeddings.

        Returns:
            loss: Circle loss value for the batch.
        """
        m = labels.size(0)
        mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)

        # Normalize embeddings and create similarity matrix
        embeddings = F.normalize(embeddings)
        sim_mat = embeddings.mm(embeddings.t())

        # Extract the similarities of positive and negative pairs from the matrix
        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]

        # Measure how far positive pairs are from ideal similarity 1 + margin
        O_p = 1 + self.margin
        alpha_p = torch.relu(O_p - pos_pair_)

        # Measure how far negative pairs are from ideal similarity of -margin
        O_n = -self.margin
        alpha_n = torch.relu(neg_pair_ - O_n)

        # Determine the margin for positives and negatives
        margin_p = 1 - self.margin
        margin_n = self.margin

        # Calculate the loss for the positives
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))

        # Calculate the loss for the negatives
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))

        # Combine the loss for the positive and negatives
        loss = torch.log(1 + (loss_p * loss_n))

        if self.debug:
            print(f"Number of positive pairs: {pos_mask.sum().item()}")
            print(f"Number of negative pairs: {neg_mask.sum().item()}")
            print(f"Shape of embeddings: {embeddings.shape}")
            print(f"Shape similarity matrix: {sim_mat.shape}")
            print(f"pos-pair shape: {pos_pair_.shape}; {pos_pair_}")
            print(f"neg-pair shape: {neg_pair_.shape}; {neg_pair_}")
            print(f"Shape alpha_p: {alpha_p.shape}; {alpha_p}")
            print(f"Shape alpha_n: {alpha_n.shape}; {alpha_n}")
            print(f"loss_p: {loss_p:.2f}")
            print(f"loss_n: {loss_n:.2f}")
            print(f"loss: {loss:.2f}")
        return loss