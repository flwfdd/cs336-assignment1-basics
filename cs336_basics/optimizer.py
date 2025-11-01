import math
from typing import Callable, Iterable, Optional

import torch
from jaxtyping import Float, Int


def cross_entropy(
    pred_logits: Float[torch.Tensor, "batch_size vocab_size"],
    targets: Int[torch.Tensor, "batch_size"],
) -> Float[torch.Tensor, ""]:
    x = pred_logits - pred_logits.max(dim=-1, keepdim=True).values
    x = x.exp().sum(dim=-1).log() - x[torch.arange(x.shape[0]), targets]
    return torch.mean(x)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:  # type: ignore[override]
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, weight_decay, betas, eps = (
                group["lr"],
                group["weight_decay"],
                group["betas"],
                group["eps"],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                g = p.grad.data
                m = betas[0] * m + (1 - betas[0]) * g
                v = betas[1] * v + (1 - betas[1]) * g**2
                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def get_lr_cosine_schedule(
    t: int, lr_max: float, lr_min: float, T_w: int, T_c: int
) -> float:
    if t < T_w:
        return t / T_w * lr_max
    if t <= T_c:
        return lr_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (
            lr_max - lr_min
        )
    return lr_min


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += (p.grad.data**2).sum().item()
    total_norm = total_norm ** (1.0 / 2)

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= clip_coef
