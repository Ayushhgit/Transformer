import torch
import torch.nn as nn
from .attention import ScaledDotProductAttention

class MultiHeadAttetion(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        def split_heads(x):
            return x.view(B, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        q, k, v = map(split_heads, (self.q_linear(q), self.k_linear(k), self.v_linear(v)))
        out, atten = self.attention(q,k,v,mask)
        out = out.permute(0,2,1,3).contiguous().view(B, -1, self.d_model)
        return self.out(out), atten