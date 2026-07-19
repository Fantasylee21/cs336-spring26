"""Text generation from a trained Transformer language model.

Supports temperature scaling and top-p (nucleus) sampling.
"""

import torch


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
    end_token: str = "<|endoftext|>",
) -> str:
    """Generate a completion for a given prompt using the language model.

    Args:
        model: A transformer_lm instance (already on the target device).
        tokenizer: A Tokenizer instance with encode/decode methods.
        prompt: The input text to complete.
        max_tokens: Maximum number of tokens to generate before stopping.
        temperature: Softmax temperature (> 0). Lower = more deterministic.
        top_p: Nucleus sampling threshold in (0, 1]. 1.0 = no filtering.
        device: Torch device string.
        end_token: Special token string that signals end of generation.

    Returns:
        The prompt concatenated with the generated completion.
    """
    model.eval()

    # Encode the prompt
    prompt_ids = tokenizer.encode(prompt)

    # Find the end-of-text token ID
    end_id = None
    if end_token in tokenizer.special_set:
        end_id = tokenizer.bytes_to_id[end_token.encode("utf-8")]

    generated_ids = []
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        # Forward pass — only need logits at the last position
        logits = model(context)          # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]   # (vocab_size,)

        # Temperature scaling
        if temperature > 0 and temperature != 1.0:
            next_logits = next_logits / temperature

        # Convert to probabilities
        probs = torch.softmax(next_logits, dim=-1)

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            probs = _nucleus_filter(probs, top_p)

        # Sample from the filtered distribution
        next_id = torch.multinomial(probs, num_samples=1).item()

        # Stop if we hit the end token
        if end_id is not None and next_id == end_id:
            break

        generated_ids.append(next_id)
        context = torch.cat(
            [context, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=-1,
        )

    full_ids = prompt_ids + generated_ids
    return tokenizer.decode(full_ids)


def _nucleus_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Filter a probability distribution to keep only the nucleus (top-p) tokens.

    Tokens are kept until cumulative probability mass exceeds *p*.  All
    other tokens are zeroed and the distribution is re-normalized.
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens while cumulative mass <= p; always keep at least the top token
    keep_mask = cumulative <= p
    keep_mask[0] = True

    filtered = torch.zeros_like(probs)
    filtered[sorted_indices[keep_mask]] = sorted_probs[keep_mask]
    return filtered / filtered.sum()
