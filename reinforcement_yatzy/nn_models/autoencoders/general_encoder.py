from abc import abstractmethod

import torch
from torch import nn


class GeneralEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        ...
