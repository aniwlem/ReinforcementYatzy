from itertools import permutations
import pytest

import numpy as np
import torch
from torch import nn

from reinforcement_yatzy.nn_models.xvariant_mlp.base_models.equivariant_layer import EquivariantLayer
from reinforcement_yatzy.nn_models.xvariant_mlp.pool_type_enum import PoolType


class TestEquivariantLayer:
    n_elems = 5
    embed_dim = 7
    n_input_channels = 3
    n_output_channels = 11

    @pytest.fixture
    def avg_layer(self):
        return EquivariantLayer(
            n_elems=self.n_elems,
            embed_dim=self.embed_dim,
            n_input_channels=self.n_input_channels,
            n_output_channels=self.n_output_channels,
            pool_type=PoolType.AVG,
        )

    @pytest.fixture
    def maxpool_layer(self):
        return EquivariantLayer(
            n_elems=self.n_elems,
            embed_dim=self.embed_dim,
            n_input_channels=self.n_input_channels,
            n_output_channels=self.n_output_channels,
            pool_type=PoolType.MAX,
        )

    @pytest.fixture
    def single_embed_dim_layer(self):
        return EquivariantLayer(
            n_elems=self.n_elems,
            embed_dim=1,
            n_input_channels=self.n_input_channels,
            n_output_channels=self.n_output_channels,
            pool_type=PoolType.AVG,
        )

    @pytest.fixture
    def single_channel_layer(self):
        return EquivariantLayer(
            n_elems=self.n_elems,
            embed_dim=self.embed_dim,
            n_input_channels=1,
            n_output_channels=self.n_output_channels,
            pool_type=PoolType.AVG,
        )

    @pytest.fixture
    def single_channel_and_embed_dim_layer(self):
        return EquivariantLayer(
            n_elems=self.n_elems,
            embed_dim=1,
            n_input_channels=1,
            n_output_channels=self.n_output_channels,
            pool_type=PoolType.AVG,
        )

    @pytest.fixture
    def model_list(self, avg_layer,  maxpool_layer):
        return [avg_layer,  maxpool_layer]

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_single_forward(self, batch_size: int, model_list: list[nn.Module]):
        batch = torch.rand([
            batch_size,
            self.n_input_channels,
            self.n_elems,
            self.embed_dim,

        ])

        for curr_layer in model_list:
            assert list(curr_layer(batch).shape) == [
                batch_size,
                self.n_output_channels,
                self.n_elems,
                self.embed_dim,
            ]

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_single_equivariance(self, batch_size: int, model_list: list[nn.Module]):
        batch = torch.rand([
            batch_size,
            self.n_input_channels,
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

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_single_dice_embed_dim(self, batch_size: int, single_embed_dim_layer: EquivariantLayer):
        batch = torch.rand([
            batch_size,
            self.n_input_channels,
            self.n_elems,
            1,

        ])

        assert list(single_embed_dim_layer(batch).shape) == [
            batch_size,
            self.n_output_channels,
            self.n_elems,
            1,
        ]

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_single_channel_dim(self, batch_size: int, single_channel_layer: EquivariantLayer):
        batch = torch.rand([
            batch_size,
            1,
            self.n_elems,
            self.embed_dim

        ])

        assert list(single_channel_layer(batch).shape) == [
            batch_size,
            self.n_output_channels,
            self.n_elems,
            self.embed_dim,
        ]

    @pytest.mark.parametrize('batch_size', range(1, 10))
    def test_single_channel_and_dim_dim(self, batch_size: int, single_channel_and_embed_dim_layer: EquivariantLayer):
        batch = torch.rand([
            batch_size,
            1,
            self.n_elems,
            1,

        ])
        print(list(single_channel_and_embed_dim_layer(batch).shape))

        assert list(single_channel_and_embed_dim_layer(batch).shape) == [
            batch_size,
            self.n_output_channels,
            self.n_elems,
            1,
        ]
