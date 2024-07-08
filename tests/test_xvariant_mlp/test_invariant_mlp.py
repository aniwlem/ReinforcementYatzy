from itertools import permutations
import pytest

import numpy as np
import torch
from torch import nn
from reinforcement_yatzy.nn_models.xvariant_mlp.base_models.invariant_mlp import InvariantMLP, InvariantPoolingParams
from reinforcement_yatzy.nn_models.xvariant_mlp.pool_type_enum import PoolType


class TestEquivariantLayer:
    n_elems = 5
    embed_dim = 7
    mlp_channels = [11, 13, 17]

    @pytest.fixture
    def avg_mlp(self):
        return InvariantMLP(
            n_elems=self.n_elems,
            embed_dim=self.embed_dim,
            mlp_channels=self.mlp_channels,
            mlp_pool_type=PoolType.AVG,
            invarintifier_pool_types=InvariantPoolingParams(
                embed_pooling=PoolType.AVG,
                elem_pooling=PoolType.AVG,
            )
        )

    @pytest.fixture
    def maxpool_mlp(self):
        return InvariantMLP(
            n_elems=self.n_elems,
            embed_dim=self.embed_dim,
            mlp_channels=self.mlp_channels,
            mlp_pool_type=PoolType.MAX,
            invarintifier_pool_types=InvariantPoolingParams(
                embed_pooling=PoolType.MAX,
                elem_pooling=PoolType.MAX,
            )
        )

    @pytest.fixture
    def single_embed_dim_mlp(self):
        return InvariantMLP(
            n_elems=self.n_elems,
            embed_dim=1,
            mlp_channels=self.mlp_channels,
            mlp_pool_type=PoolType.MAX,
            invarintifier_pool_types=InvariantPoolingParams(
                embed_pooling=PoolType.MAX,
                elem_pooling=PoolType.MAX,
            )
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
            ]

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_invariance(self, batch_size: int, model_list: list[nn.Module]):
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
                    (curr_layer(batch[:, :, perm, :]) == results).numpy
                )

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_single_dice_embed_dim(self, batch_size: int, single_embed_dim_mlp: InvariantMLP):
        batch = torch.rand([
            batch_size,
            self.mlp_channels[0],
            self.n_elems,
            1,

        ])

        assert list(single_embed_dim_mlp(batch).shape) == [
            batch_size,
            self.mlp_channels[-1],
        ]
