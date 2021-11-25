"""Simple transformer for text classification."""
import torch
from torch.nn import (
    CrossEntropyLoss,
    Embedding,
    Linear,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from ..config import Config
from .base import ClassifierBase


class SmallTransformer(ClassifierBase):
    """A tiny BERT-like transformer for text classification."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_classes: int,
        config: Config,
        model_dims: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
    ):
        """Initialize the model.

        Args:
            vocab_size: The size of the vocabulary of tokens
            max_seq_len: The maximum possible sequence length
            num_classes: The total number of classes
            config: The hyper-param config
            model_dims: The dimensions for the embeddings and token features
            num_heads: The number of multi-headed attention heads
            num_layers: The number of transformer layers
        """
        super().__init__(config, CrossEntropyLoss())

        self.word_embed = Embedding(vocab_size, model_dims)
        self.pos_embed = Embedding(max_seq_len, model_dims)

        # The encoder of the original Transformer
        layer = TransformerEncoderLayer(
            model_dims, num_heads, batch_first=True
        )
        self.encoder = TransformerEncoder(layer, num_layers)
        self.fc = Linear(model_dims, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get the classification logits for these inputs."""
        word_embeddings = self.word_embed(inputs)
        pos_embeddings = self.pos_embed.weight[: inputs.shape[1]]
        embeddings = word_embeddings + pos_embeddings.unsqueeze(0)
        outputs = self.encoder(embeddings)
        pooled = outputs.mean(1)
        return self.fc(pooled)
