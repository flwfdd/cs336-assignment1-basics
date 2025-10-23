import math

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float


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
