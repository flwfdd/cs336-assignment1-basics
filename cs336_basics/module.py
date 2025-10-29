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


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct the RMSNorm module.
        """
        super().__init__()
        self.eps = eps
        self.gain: Float[torch.Tensor, "d_model"] = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.gain

        return result.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.silu = SiLU()
        self.d_ff = math.floor(8 / 3 * d_model / 64) * 64
        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        return self.W2.forward(self.silu(self.W1.forward(x)) * self.W3.forward(x))


class RoPE(nn.Module):
    cos: Float[torch.Tensor, "max_seq_len d_k_half"]
    sin: Float[torch.Tensor, "max_seq_len d_k_half"]

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        """
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        """
        super().__init__()
        thetas = einsum(
            torch.arange(max_seq_len),
            torch.pow(theta, -torch.arange(0, d_k, 2) / d_k),
            "index, theta -> index theta",
        )
        cos = torch.cos(thetas).to(device)  # max_seq_len d_k/2
        sin = torch.sin(thetas).to(device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Float[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        x_even = x[..., ::2]  # ... seq_len d_k/2
        x_odd = x[..., 1::2]
        cos = self.cos[token_positions]  # ... seq_len d_k/2
        sin = self.sin[token_positions]
        result = torch.empty_like(x)  # ... seq_len d_k
        result[..., ::2] = x_even * cos - x_odd * sin
        result[..., 1::2] = x_even * sin + x_odd * cos
        return result
