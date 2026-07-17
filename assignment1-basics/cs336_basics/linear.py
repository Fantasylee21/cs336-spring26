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
        nn.init.trunc_normal_(self.weight, mean=0.0, std=math.sqrt(2 / (in_features + out_features)))

    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    