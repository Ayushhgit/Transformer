import math
import torch
import torch.nn as nn
from .transformer_block import TransformerDecoderBlock
from .positional_encoding import PositionalEncoding


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
        TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
        for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model


    def forward(self, x, mask=None):
        x = self.embed(x) * math.sqrt(self.d_model)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(self.norm(x))