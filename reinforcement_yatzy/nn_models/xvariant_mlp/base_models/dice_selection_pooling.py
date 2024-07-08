from dataclasses import dataclass
import torch
from torch import nn

from reinforcement_yatzy.nn_models.xvariant_mlp.pool_type_enum import PoolType


@dataclass
class DiceSelectionPoolingTypes:
    channel_pooling: PoolType
    embed_pooling: PoolType


class DiceSelectionPooling(nn.Module):
    '''
    Class that applies two successive pooling layers to the output of a
    permutation equivariant dice selector with multiple channels and vectorial
    embeddings.
    '''

    def __init__(
        self,
        embed_dim: int,
        n_channels: int,
        channel_pooling: PoolType,
        embed_pooling: PoolType,
    ) -> None:
        super().__init__()

        self.channel_pooling = channel_pooling.layer(n_channels)
        self.embed_pooling = embed_pooling.layer(embed_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        '''
        inputs: shape[batch_size, input_channels, n_elems, embed_dim]

        outputs: shape[batch_size, n_elems]
        '''
        if len(batch.shape) == 3:
            batch.unsqueeze(0)

        batch_size, n_channels, n_dice, embed_dim = batch.shape
        # Pool away embedding dimension
        embed_batch = batch.reshape(
            [batch_size * n_channels, n_dice, embed_dim])
        embed_pooled = self.embed_pooling(embed_batch).reshape([
            batch_size, n_channels, n_dice
        ])

        # need to get channel dim last
        channel_batch = embed_pooled.permute([0, 2, 1])
        # Pool away channel dimension
        channel_pooled = self.channel_pooling(channel_batch).squeeze()
        return channel_pooled
