import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for 2D coordinates.
    Args:
        dim (int): Dimension of the positional encoding.

    Returns:
        torch.Tensor: Positional encoding of shape (batch_size, 2 * dim).
    """

    def __init__(self, dim) -> None:
        super(PositionalEncoding, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, dim), requires_grad=True)

    def forward(self, coord):
        coord = torch.unsqueeze(coord, dim=-1)
        freqs = torch.matmul(coord, self.weights) * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim=-1).view(coord.shape[0], -1)
