import torch.nn as nn
from collections import OrderedDict

from .modules import PositionalEncoding


class NeRF(nn.Module):
    """NeRF model.
    Args:
        config (ArgumentParser): Config object
        inp_size (int): Input size of the positional encoding.

    Returns:
        torch.Tensor: Color of shape (batch_size, 3).
    """

    def __init__(self, config, inp_size) -> None:
        super(NeRF, self).__init__()
        self.config = config

        if config.activation == "relu":
            self.act = nn.ReLU()
        elif config.activation == "relu6":
            self.act = nn.ReLU6()
        elif config.activation == "elu":
            self.act = nn.ELU()

        if config.learnable_positional_encoding:
            self.encode = PositionalEncoding(config.max_freq_exp)

        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("hidden_1", nn.Linear(inp_size, config.hidden_dim)),
                    ("activation", self.act),
                    ("hidden_2", nn.Linear(config.hidden_dim, config.hidden_dim)),
                    ("activation", self.act),
                    ("final_out", nn.Linear(config.hidden_dim, 3)),
                ]
            )
        )

    def forward(self, pixel_coord):
        if self.config.learnable_positional_encoding:
            pixel_coord = self.encode(pixel_coord)
        return self.net(pixel_coord)
