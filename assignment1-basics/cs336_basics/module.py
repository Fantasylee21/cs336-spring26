import torch.nn as nn
import torch
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
        nn.init.ones_(self.weight)

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / RMS * self.weight).to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if dtype is None:
            dtype = torch.float32
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        x1 = self.linear1(x)
        x3 = self.linear3(x)
        return self.linear2(torch.nn.functional.silu(x1) * x3)

class RoPEEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.freq_arrange = 1 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        self.register_buffer("inv_freq", self.freq_arrange, persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate consecutive dimension pairs: [x0,x1,x2,x3,...] -> [-x1,x0,-x3,x2,...]"""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"

        freqs = torch.einsum("...s,j->...sj", token_positions.float(), self.inv_freq)
        freqs = freqs.repeat_interleave(2, dim=-1)
        cos, sin = freqs.cos(), freqs.sin()
        x_rotated = (x * cos) + (self.rotate_half(x) * sin)
        return x_rotated
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax implementation."""
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = query.shape[-1]
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output

class Multihead_Self_Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_t = d_model // num_heads

        self.query_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.key_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.value_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_linear = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None:
            self.theta = theta
            self.max_seq_len = max_seq_len
            self.rope = RoPEEmbedding(theta, self.d_t, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        q = self.query_linear(x)
        k = self.key_linear(x)
        v = self.value_linear(x)

        q = q.view(*q.shape[:-2], seq_len, self.num_heads, self.d_t).transpose(-3, -2)
        k = k.view(*k.shape[:-2], seq_len, self.num_heads, self.d_t).transpose(-3, -2)
        v = v.view(*v.shape[:-2], seq_len, self.num_heads, self.d_t).transpose(-3, -2)

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)).unsqueeze(0)

        attention_output = scaled_dot_product_attention(q, k, v, mask)

        attention_output = attention_output.transpose(-3, -2).contiguous().view(*attention_output.shape[:-3], seq_len, self.d_model)
        output = self.out_linear(attention_output)
        return output