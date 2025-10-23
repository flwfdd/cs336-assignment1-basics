import math

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float, Int


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        linear transformation module.
        """
        super().__init__()
        self.W: Float[torch.Tensor, "in_features out_features"] = nn.Parameter(
            torch.empty((in_features, out_features), device=device, dtype=dtype)
        )
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, 0, std, -3 * std, 3 * std)

    def forward(
        self, x: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        """
        Apply the linear transformation to the input.
        """
        return einsum(x, self.W, "... d_in, d_in d_out -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an embedding module.

        num_embeddings: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors, i.e., d_model
        """
        super().__init__()
        self.embedding: Float[torch.Tensor, "num_embeddings embedding_dim"] = (
            nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
            )
        )
        nn.init.trunc_normal_(self.embedding, 0, 1, -3, 3)

    def forward(
        self, token_ids: Int[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "... embedding_dim"]:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.embedding[token_ids]
