import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from mup import MuReadout

class TransformerModel(nn.Module):

    def __init__(self, n_tokens: int, d_model: int, n_heads: int,
                 n_layers: int, dropout: float = 1.0, mup: bool = True):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_model, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(n_tokens, d_model)
        if mup:
            self.decoder = MuReadout(self.d_model, self.n_tokens, bias=False)
        else:
            self.decoder = nn.Linear(self.d_model, self.n_tokens, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, feat_dim]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, n_tokens]``
        """
        x = self.encoder(x) * self.d_model # MuP uses d_model not sqrt(d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, is_causal=True)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)