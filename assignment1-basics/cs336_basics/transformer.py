import torch.nn as nn
import torch
from cs336_basics.module import Linear, RMSNorm, SwiGLU, Multihead_Self_Attention, Embedding


class transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len: int | None = None, theta: float | None = None, device=None, dtype=None):
        super().__init__()
        self.attention = Multihead_Self_Attention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        token_positions = torch.arange(x.shape[-2], device=x.device).unsqueeze(0)
        x = x + self.attention(self.norm1(x), token_positions)
        x = x + self.ffn(self.norm2(x))
        return x
    

class transformer_lm(nn.Module):
    def __init__(self , vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta: float | None = None, device=None, dtype=None):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            transformer_block(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output_linear = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.output_linear(x)