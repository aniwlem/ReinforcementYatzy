from torch import nn

from reinforcement_yatzy.nn_models.xvariant_mlp.base_models.equivariant_layer import EquivariantLayer

from reinforcement_yatzy.nn_models.xvariant_mlp.pool_type_enum import PoolType


class EquivariantMLP(nn.Module):
    def __init__(
        self,
        n_elems: int,
        embed_dim: int,
        mlp_channels: list[int],
        pool_type: PoolType,
    ):
        super().__init__()

        mlp_layers = []

        for i in range(len(mlp_channels) - 1):
            mlp_layers.extend([
                EquivariantLayer(
                    n_elems,
                    embed_dim,
                    mlp_channels[i],
                    mlp_channels[i + 1],
                    pool_type,
                ),
                nn.ReLU(),
            ])

        self.mlp = nn.ModuleList(mlp_layers)

    def forward(self, batch):
        for layer in self.mlp:
            batch = layer(batch)

        return batch
