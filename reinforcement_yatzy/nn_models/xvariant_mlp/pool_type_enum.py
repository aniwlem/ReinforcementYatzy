from enum import Enum

from torch import nn


class PoolType(Enum):
    AVG = 'avg'
    MAX = 'max'

    def layer(self, dim: int) -> nn.Module:
        if self == PoolType.AVG:
            return nn.AvgPool1d(dim)
        elif self == PoolType.MAX:
            return nn.MaxPool1d(dim)
        else:
            raise ValueError('Unsupported pooling type')
