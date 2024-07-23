import torch
from torch import nn

from reinforcement_yatzy.nn_models.autoencoders.mlp_scoreboard_autoencoder import MLPScoreboardEncoder
from reinforcement_yatzy.nn_models.xvariant_mlp.base_models.invariant_mlp import InvariantMLP, InvariantPoolingParams
from reinforcement_yatzy.nn_models.xvariant_mlp.pool_type_enum import PoolType


class EntrySelector(nn.Module):
    def __init__(
        self,
        n_dice: int,
        n_entries: int,
        dice_pre_mlp_channels: list[int],
        scoreboard_encoder: MLPScoreboardEncoder,
        mlp_dims: list[int],
        dice_pre_mlp_pool_type: PoolType = PoolType.AVG,
        dice_invarintifier_pool_types: InvariantPoolingParams =
            InvariantPoolingParams(
                PoolType.AVG,
                PoolType.AVG,
        ),
    ) -> None:
        super().__init__()

        self.dice_invariantifier = InvariantMLP(
            n_elems=n_dice,
            embed_dim=1,  # just the dice values
            mlp_channels=[1, *dice_pre_mlp_channels],
            mlp_pool_type=dice_pre_mlp_pool_type,
            invarintifier_pool_types=dice_invarintifier_pool_types,
        )
        self.scoreboard_encoder = scoreboard_encoder

        mlp_layers = [
            nn.Linear(
                self.scoreboard_encoder.latent_dim + dice_pre_mlp_channels[-1],
                mlp_dims[0],
            ),
            nn.ReLU(),
        ]

        for i in range(len(mlp_dims) - 1):
            mlp_layers.extend([
                nn.Linear(
                    mlp_dims[i],
                    mlp_dims[i + 1],
                ),
                nn.ReLU(),
            ])

        mlp_layers.extend([
            nn.Linear(
                mlp_dims[-1],
                n_entries,
            ),
            nn.ReLU(),
        ])

        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(self, dices: torch.Tensor, scoreboards: torch.Tensor) -> torch.Tensor:
        '''
        inputs:
        dice: shape[batch_size, n_dice]
        scoreboards: shape[batch_size, n_entries]

        output: shape[batch_size, n_entries]
        '''

        # add channel and embed dims
        batch_size, n_dice = dices.shape
        dice_batch = dices.reshape([batch_size, 1, n_dice, 1])
        invariant_dice = self.dice_invariantifier(dice_batch)

        # the encoder will add channel and embed dims itself
        scoreboard_embeddings = self.scoreboard_encoder(scoreboards)
        batch = torch.concat([
            invariant_dice,
            scoreboard_embeddings,
        ], dim=1)

        for layer in self.mlp_layers:
            batch = layer(batch)

        return batch
