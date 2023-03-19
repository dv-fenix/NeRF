import torch.nn as nn
from collections import OrderedDict


class NeRF(nn.Module):
    def __init__(self, config, inp_size) -> None:
        super(NeRF, self).__init__()

        if config.activation == "relu":
            self.act = nn.ReLU()
        elif config.activation == "relu6":
            self.act = nn.ReLU6()
        elif config.activation == "elu":
            self.act = nn.ELU()

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
        return self.net(pixel_coord)
