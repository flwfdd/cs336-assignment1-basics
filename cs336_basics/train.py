from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, Int


def get_batch(
    x: Int[npt.NDArray, "length"],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[
    Int[torch.Tensor, "batch_size context_length"],
    Int[torch.Tensor, "batch_size context_length"],
]:
    indices = np.random.randint(0, x.shape[0] - context_length, size=(batch_size,))
    inputs = np.stack([x[i : i + context_length] for i in indices])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in indices])
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs_tensor, targets_tensor
