from itertools import permutations
import pytest

import numpy as np
import torch
from torch import nn
from reinforcement_yatzy.nn_models.xvariant_mlp.base_models.equivariant_mlp import EquivariantMLP
from reinforcement_yatzy.nn_models.xvariant_mlp.pool_type_enum import PoolType


class TestEquivariantLayer:
    n_elems = 5
    embed_dim = 7
    mlp_channels = [11, 13, 17]

    @pytest.fixture
    def avg_mlp(self):
        return EquivariantMLP(
            n_elems=self.n_elems,
            embed_dim=self.embed_dim,
            mlp_channels=self.mlp_channels,
            pool_type=PoolType.AVG,
        )

    @pytest.fixture
    def maxpool_mlp(self):
        return EquivariantMLP(
            n_elems=self.n_elems,
            embed_dim=self.embed_dim,
            mlp_channels=self.mlp_channels,
            pool_type=PoolType.MAX,
        )

    @pytest.fixture
    def model_list(self, avg_mlp,  maxpool_mlp):
        return [avg_mlp,  maxpool_mlp]

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_forward(self, batch_size: int, model_list: list[nn.Module]):
        batch = torch.rand([
            batch_size,
            self.mlp_channels[0],
            self.n_elems,
            self.embed_dim,
        ])

        for curr_layer in model_list:
            assert list(curr_layer(batch).shape) == [
                batch_size,
                self.mlp_channels[-1],
                self.n_elems,
                self.embed_dim,
            ]

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_equivariance(self, batch_size: int, model_list: list[nn.Module]):
        batch = torch.rand([
            batch_size,
            self.mlp_channels[0],
            self.n_elems,
            self.embed_dim,
        ])
        for curr_layer in model_list:
            results = curr_layer(batch)

            perms = list(permutations(range(self.n_elems)))
            for perm in perms:
                assert np.all(
                    (curr_layer(batch[:, :, perm, :])
                        == results[:, :, perm, :]).numpy
                )