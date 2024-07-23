import pytest
from itertools import permutations

import numpy as np
import torch

from reinforcement_yatzy.nn_models.xvariant_mlp.base_models.invariant_mlp import InvariantPoolingParams
from reinforcement_yatzy.nn_models.xvariant_mlp.selection_models.entry_selector import EntrySelector
from reinforcement_yatzy.nn_models.autoencoders.mlp_scoreboard_autoencoder import MLPScoreboardEncoder

from reinforcement_yatzy.nn_models.xvariant_mlp.pool_type_enum import PoolType


class TestEntrySelector:
    n_dice = 3
    n_entries = 13
    dice_embed_dim = 2
    scoreboard_embed_dim = 5

    @pytest.fixture
    def encoder(self):
        return MLPScoreboardEncoder(
            n_entries=self.n_entries,
            mlp_dims=[3, 5, 7],
            latent_dim=self.scoreboard_embed_dim
        )

    @pytest.fixture
    def entry_selector(self, encoder: MLPScoreboardEncoder):
        return EntrySelector(
            n_dice=self.n_dice,
            n_entries=self.n_entries,
            dice_pre_mlp_channels=[6, 2, 4],
            dice_pre_mlp_pool_type=PoolType.AVG,
            dice_invarintifier_pool_types=InvariantPoolingParams(
                embed_pooling=PoolType.AVG,
                elem_pooling=PoolType.AVG,
            ),
            scoreboard_encoder=encoder,
            mlp_dims=[16, 32, 16, 8, self.n_entries],
        )

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_forward(self, entry_selector: EntrySelector, batch_size: int):
        dices = torch.rand([batch_size, self.n_dice])
        entries = torch.rand([batch_size, self.n_entries])

        assert list(entry_selector(dices, entries).shape) == [
            batch_size,
            self.n_entries
        ]

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_invariance(self, entry_selector: EntrySelector, batch_size: int):
        dices = torch.rand([batch_size, self.n_dice])
        entries = torch.rand([batch_size, self.n_entries])

        results = entry_selector(dices, entries)
        perms = list(permutations(range(self.n_dice)))

        for perm in perms:
            assert np.all(
                (entry_selector(dices[:, perm], entries) == results).numpy
            )
