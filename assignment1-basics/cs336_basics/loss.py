import torch
import torch.nn as nn
from typing import Iterable

def cross_entropy_loss(logits, targets):
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values

    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return loss.mean()

def perplexity(loss: torch.Tensor):
    return torch.exp(loss)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total_norm = torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in parameters if p.grad is not None))
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
