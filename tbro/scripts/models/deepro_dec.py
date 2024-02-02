from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

class deepRODecoder(pl.LightningModule):
    def __init__(
        self, input_dim: int
    ):
        super().__init__()
        self.linear_tf = nn.Linear(input_dim,6)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.linear_tf(x)
