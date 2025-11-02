import os
import typing

import numpy as np
import numpy.typing as npt
import torch
import wandb
from jaxtyping import Int
from module import TransformerLM
from optimizer import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping


def get_batch(
    x: Int[npt.NDArray, "length"],
    batch_size: int,
    context_length: int,
    device: str | torch.device,
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


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]
    return iteration


class TrainConfig(typing.TypedDict):
    device: torch.device
    dtype: torch.dtype
    # Transformer LM
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    # optimizer
    lr: float
    lr_min: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    max_grad_norm: float
    # data
    token_ids_path: str | os.PathLike
    checkpoint_dir: str | os.PathLike
    # train
    batch_size: int
    total_tokens: int
    validation_interval: int
    checkpoint_interval: int
    wandb_project: str
    wandb_name: str


TinyStoriesConfig = TrainConfig(
    device=torch.device("cuda"),
    dtype=torch.float32,
    # Transformer LM
    vocab_size=10000,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    rope_theta=10000,
    # optimizer
    lr=3e-4,
    lr_min=3e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    max_grad_norm=1.0,
    # data
    token_ids_path="../data/TinyStoriesV2-GPT4-train/token_ids.npy",
    checkpoint_dir="../data/checkpoints/tiny_stories",
    # train
    batch_size=128,
    total_tokens=327_680_000,
    validation_interval=10,
    checkpoint_interval=1000,
    wandb_project="cs336",
    wandb_name="tiny_stories_h100",
)


def train(config: TrainConfig) -> None:
    lm = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=config["device"],
        dtype=config["dtype"],
    )
    adamw = AdamW(
        lm.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
        eps=config["eps"],
    )
    token_ids = np.load(config["token_ids_path"], mmap_mode="r")
    total_steps = (
        config["total_tokens"] // config["batch_size"] // config["context_length"]
    )
    print(f"Total training steps: {total_steps}")
    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_name"],
        config={**config, "total_steps": total_steps},
    )
    for step in range(total_steps):
        inputs, targets = get_batch(
            token_ids,
            batch_size=config["batch_size"],
            context_length=config["context_length"],
            device=config["device"],
        )  # batch_size x context_length
        logits = lm(inputs)  # batch_size x context_length x vocab_size
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        adamw.zero_grad()
        loss.backward()
        gradient_clipping(lm.parameters(), config["max_grad_norm"])
        lr = get_lr_cosine_schedule(
            step,
            lr_max=config["lr"],
            lr_min=config["lr_min"],
            T_w=total_steps // 10,
            T_c=total_steps,
        )
        for param_group in adamw.param_groups:
            param_group["lr"] = lr
        adamw.step()
        wandb.log({"loss": loss.item(), "lr": lr}, step=step)
        if (step + 1) % config["validation_interval"] == 0:
            print(f"Step {step+1}: loss = {loss.item():.4f}")
        if (step + 1) % config["checkpoint_interval"] == 0:
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
            checkpoint_path = os.path.join(
                config["checkpoint_dir"], f"checkpoint_step_{step+1}.pt"
            )
            save_checkpoint(lm, adamw, step + 1, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["checkpoint_dir"], f"checkpoint_final.pt")
    save_checkpoint(
        lm,
        adamw,
        total_steps,
        checkpoint_path,
    )


if __name__ == "__main__":
    train(TinyStoriesConfig)
