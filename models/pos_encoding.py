import math
import torch
from torch import nn

"""
Implements various positional encoding strategies for sequence modeling tasks
and provides support both fixed sinusoidal encoding and learned positional embeddings.
"""

def PE1d_sincos(seq_length, dim):
    """
    Generate 1D sinusoidal positional encoding as described in "Attention Is All You Need" paper:

    https://arxiv.org/abs/1706.03762v7

    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                         -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)


class PositionEmbedding(nn.Module):
    """
    Flexible positional embedding module via (Absolute pos embedding (standard), learned).
    """
    def __init__(self, seq_length, dim, dropout, grad=False):
        super().__init__()
        self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """
        Apply positional encoding
        """
        # x.shape: bs, seq_len, feat_dim
        l = x.shape[1]
        x = x.permute(1, 0, 2) + self.embed[:l].expand(x.permute(1, 0, 2).shape)
        x = self.dropout(x.permute(1, 0, 2))
        return x

 