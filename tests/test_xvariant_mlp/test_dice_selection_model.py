import pytest
from itertools import permutations

import numpy as np
import torch

from reinforcement_yatzy.nn_models.xvariant_mlp.base_models.dice_selection_pooling import DiceSelectionPoolingTypes
from reinforcement_yatzy.nn_models.xvariant_mlp.selection_models.dice_selector import DiceSelector
from reinforcement_yatzy.nn_models.autoencoders.scoreboard_autoencoder import ScoreboardEncoder

from reinforcement_yatzy.nn_models.xvariant_mlp.pool_type_enum import PoolType


class TestDiceSelector:
    n_dice = 3
    n_entries = 13
    dice_embed_dim = 2
    scoreboard_embed_dim = 5

    @pytest.fixture
    def encoder(self):
        return ScoreboardEncoder(
            input_dim=self.n_entries,
            mlp_dims=[3, 5, 7],
            latent_dim=self.scoreboard_embed_dim
        )

    @pytest.fixture
    def dice_selector(self, encoder: ScoreboardEncoder):
        return DiceSelector(
            n_dice=self.n_dice,
            dice_embed_dim=self.dice_embed_dim,
            mlp_channels=[13, 17, 19],
            mlp_pool_type=PoolType.AVG,
            selection_pooling_types=DiceSelectionPoolingTypes(
                channel_pooling=PoolType.AVG,
                embed_pooling=PoolType.AVG
            ),
            scoreboard_encoder=encoder,
        )

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_forward(self, dice_selector: DiceSelector, batch_size: int):
        dices = torch.rand([batch_size, self.n_dice])
        entries = torch.rand([batch_size, self.n_entries])

        if batch_size == 1:
            assert list(dice_selector(dices, entries).shape) == [self.n_dice]
        else:
            assert list(dice_selector(dices, entries).shape) == [
                batch_size,
                self.n_dice
            ]

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_equivariance(self, dice_selector: DiceSelector, batch_size: int):
        dices = torch.rand([batch_size, self.n_dice])
        entries = torch.rand([batch_size, self.n_entries])

        results = dice_selector(dices, entries)
        perms = list(permutations(range(self.n_dice)))

        if batch_size == 1:
            print(dices)
            print(dices.unsqueeze(0)[0, perms[0]])
            print(entries.unsqueeze(0))
            print(dice_selector(
                dices.unsqueeze(0)[0, perms[0]],
                entries.unsqueeze(0))
            )

            for perm in perms:
                assert np.all(
                    (dice_selector(dices.unsqueeze(0)[:, perm], entries.unsqueeze(0))
                        == results.unsqueeze(0)[:, perm]).numpy
                )
        else:
            for perm in perms:
                assert np.all(
                    (dice_selector(dices[:, perm], entries)
                        == results[:, perm]).numpy
                )
