from dataclasses import asdict
import torch
from torch import nn

from reinforcement_yatzy.nn_models.xvariant_mlp.base_models.dice_selection_pooling import DiceSelectionPooling
from reinforcement_yatzy.nn_models.xvariant_mlp.base_models.equivariant_mlp import EquivariantMLP
from reinforcement_yatzy.nn_models.autoencoders.scoreboard_autoencoder import ScoreboardEncoder

from reinforcement_yatzy.nn_models.xvariant_mlp.pool_type_enum import PoolType
from reinforcement_yatzy.nn_models.xvariant_mlp.base_models.dice_selection_pooling import DiceSelectionPoolingTypes


class DiceSelector(nn.Module):
    def __init__(
        self,
        n_dice: int,
        dice_embed_dim: int,
        mlp_channels: list[int],
        mlp_pool_type: PoolType,
        selection_pooling_types: DiceSelectionPoolingTypes,
        scoreboard_encoder: ScoreboardEncoder,

    ) -> None:
        super().__init__()
        self.n_dice = n_dice

        self.mlp = EquivariantMLP(
            n_elems=n_dice,
            # +1 for throw index
            embed_dim=dice_embed_dim + scoreboard_encoder.latent_dim + 1,
            # Don't have to specify that first layer is one channel
            mlp_channels=[1, *mlp_channels],
            pool_type=mlp_pool_type,
        )

        self.selection_pool = DiceSelectionPooling(
            # +1 for throw index
            embed_dim=dice_embed_dim + scoreboard_encoder.latent_dim + 1,
            n_channels=mlp_channels[-1],
            **asdict(selection_pooling_types),
        )

        # TODO: add more layers?
        self.dice_updimensionalizer = nn.ModuleList([
            nn.Linear(1, dice_embed_dim),
            nn.ReLU(),
        ])

        self.scoreboard_encoder = scoreboard_encoder

    def _embed_dice(self, dice: torch.Tensor):
        # Must add dimension that can be updimensionalized
        dice = dice.unsqueeze(-1)
        for layer in self.dice_updimensionalizer:
            dice = layer(dice)
        return dice

    def forward(
        self,
        dices: torch.Tensor,
        scoreboards: torch.Tensor,
        throw_indices: torch.Tensor,
    ) -> torch.Tensor:
        '''
        input:
        dices: shape[batch_size, n_dice]
        scoreboards: shape[batch_size, n_entries]
        throw_indices: shape[batch_size]

        output: shape[batch_size, n_dice]
        '''
        # Of course n_dice could be 1 as well, but that will never happen
        if len(dices.shape) == 1:
            dices.unsqueeze(0)
            scoreboards.unsqueeze(0)

        batch_size = dices.shape[0]

        # [batch, n_dice, dice_embed_dim]
        dice_embeds = self._embed_dice(dices)
        # [batch, scoreboard_embed_dim]
        scoreboard_embeds = self.scoreboard_encoder(scoreboards)

        # Each dice element get the same scoreboard embedding appended at the
        # end. All equivariant vectors have the same scoreboard embeddings but
        # different dice embeddings.
        batch = torch.concat([
            dice_embeds,
            # add n_dice dimension and repeat along it
            scoreboard_embeds.unsqueeze(1).repeat([1, self.n_dice, 1]),
            # Add dice and embed_dims and repeat along dice dimension
            throw_indices.reshape(batch_size, 1, 1).repeat(
                [1, self.n_dice, 1]
            ),
        ],
            dim=-1)

        batch = batch.unsqueeze(1)  # Add channel dim
        mlped_batch = self.mlp(batch)

        # Now the results have shape [batch_size, n_channels, n_dice, n_embeds]
        # since the model only needs one value per dice the 1:st and 3:rd dimensions
        # must be pooled over.  NOTE: Could use NN instead, why not?
        results = self.selection_pool(mlped_batch)

        if len(results.shape) == 1:
            results = results.unsqueeze(0)

        return results
