import torch
from module import TransformerLM, softmax
from tokenizer import Tokenizer


def decode(
    tokenizer: Tokenizer,
    lm: TransformerLM,
    prompt: str,
    stop_token: str,
    context_length: int,
    temperature: float,
    top_p: float,
    device: str | torch.device,
) -> str:
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    stop_token_id = tokenizer.encode(stop_token)[0]

    # Generate new tokens
    with torch.no_grad():
        for _ in range(context_length - input_ids.shape[1]):
            logits = lm(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = softmax(logits, dim=-1)

            # Apply top-p sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = probs.sort(descending=True)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                cutoff = cumulative_probs > top_p
                cutoff[:, 0] = False  # Ensure at least one token is kept
                sorted_probs[cutoff] = 0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_indices.gather(
                    -1, torch.multinomial(sorted_probs, 1)
                )
            else:
                next_token = torch.multinomial(probs, 1)

            if next_token.item() == stop_token_id:
                break
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode the generated tokens
    output = tokenizer.decode(input_ids[0].cpu().tolist())
    return output


if __name__ == "__main__":
    import os

    from train import TinyStoriesConfig

    # Load tokenizer and model (paths are examples)
    tokenizer = Tokenizer.from_files(
        "../data/TinyStoriesV2-GPT4-train/bpe_vocab.pkl",
        "../data/TinyStoriesV2-GPT4-train/bpe_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    config = TinyStoriesConfig
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
    checkpoint = torch.load(
        "../data/checkpoints/tiny_stories/checkpoint_final.pt",
        map_location=config["device"],
    )
    lm.load_state_dict(checkpoint["model"])
    lm.eval()

    prompt = "Once upon a time"
    stop_token = "<|endoftext|>"
    generated_text = decode(
        tokenizer,
        lm,
        prompt,
        stop_token,
        context_length=200,
        temperature=1.0,
        top_p=0.9,
        device=config["device"],
    )
    print(generated_text)
