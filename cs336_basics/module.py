import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from einops import einsum, rearrange
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


def silu(
    x: Float[torch.Tensor, "..."],
) -> Float[torch.Tensor, "..."]:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_ff = d_ff
        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        return self.W2.forward(silu(self.W1.forward(x)) * self.W3.forward(x))


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


def softmax(x: Float[torch.Tensor, " ..."], dim: int):
    x -= x.max(dim, keepdim=True).values  # make max to 0
    exp = x.exp()
    return exp / exp.sum(dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "batch_size ... queries d_k"],
    K: Float[torch.Tensor, "batch_size ... keys d_k"],
    V: Float[torch.Tensor, "batch_size ... values d_v"],
    mask: Float[torch.Tensor, "queries keys"] | None = None,
) -> Float[torch.Tensor, "batch_size ... queries d_v"]:
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    return einsum(
        softmax(scores / math.sqrt(K.shape[-1]), -1),
        V,
        "... queries keys, ... keys d_v -> ... queries d_v",
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pos_encoder: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.num_heads = num_heads
        self.pos_encoder = pos_encoder
        self.W_q = Linear(
            d_model, d_model, device=device, dtype=dtype
        )  # d_model -> h*d_q
        self.W_k = Linear(
            d_model, d_model, device=device, dtype=dtype
        )  # d_model -> h*d_k
        self.W_v = Linear(
            d_model, d_model, device=device, dtype=dtype
        )  # d_model -> h*d_v
        self.W_o = Linear(
            d_model, d_model, device=device, dtype=dtype
        )  # h*d_v -> d_model

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size ... seq_len d_model"],
        token_positions: Int[torch.Tensor, " ... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "batch_size ... seq_len d_model"]:
        seq_len = x.shape[-2]
        Q: Float[torch.Tensor, "batch_size ... seq_len h*d_q"] = self.W_q(x)
        K: Float[torch.Tensor, "batch_size ... seq_len h*d_k"] = self.W_k(x)
        V: Float[torch.Tensor, "batch_size ... seq_len h*d_v"] = self.W_v(x)
        # split heads
        Qh: Float[torch.Tensor, "batch_size ... h seq_len d_q"] = rearrange(
            Q, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads
        )
        Kh: Float[torch.Tensor, "batch_size ... h seq_len d_k"] = rearrange(
            K, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads
        )
        Vh: Float[torch.Tensor, "batch_size ... h seq_len d_v"] = rearrange(
            V, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads
        )
        # apply positional encoding if any
        if self.pos_encoder is not None and token_positions is not None:
            Qh = self.pos_encoder(Qh, token_positions)
            Kh = self.pos_encoder(Kh, token_positions)
        # create mask for causal attention
        mask = (
            torch.triu(
                torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
                diagonal=1,
            )
            == 0
        )
        # apply attention and merge heads
        out: Float[torch.Tensor, "batch_size ... seq_len h*d_v"] = rearrange(
            scaled_dot_product_attention(Qh, Kh, Vh, mask),
            "... h seq_len d -> ... seq_len (h d)",
        )
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.rope = RoPE(rope_theta, d_model // num_heads, max_seq_len, device)
        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)
        self.mha = MultiHeadSelfAttention(
            d_model, num_heads, pos_encoder=self.rope, device=device, dtype=dtype
        )
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.token_positions: Int[torch.Tensor, "max_seq_len"] = torch.arange(
            max_seq_len, device=device
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len d_model"],
    ) -> Float[torch.Tensor, "batch_size seq_len d_model"]:
        # Multi-Head Attention with RoPE
        x_residual = x
        x = self.norm1(x)
        x = self.mha(x, self.token_positions[: x.shape[-2]])
        x = x + x_residual

        # Feed-Forward Network with SwiGLU
        x_residual = x
        x = self.norm2(x)
        x = self.swiglu(x)
        x = x + x_residual

        return x


class TransformerLM(
    nn.Module,
):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, context_length, rope_theta, device, dtype
                )
                for _ in range(num_layers)
            ]
        )
        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)
        self.norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_embedding = Linear(d_model, vocab_size, device, dtype)

    def forward(
        self,
        token_ids: Int[torch.Tensor, "batch_size sequence_length"],
    ) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        x = self.token_embedding(token_ids)
        for block in self.layers:
            x = block(x)
        x = self.norm_final(x)
        x = self.output_embedding(x)
        return x
