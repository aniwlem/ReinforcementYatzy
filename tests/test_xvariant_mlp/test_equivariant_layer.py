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

    # @pytest.mark.parametrize('batch_size', range(1, 10))
    # def test_equivariant_arb_out_dim_forward(self, batch_size: int):
    #     out_dim = 6
    #     layer = EquivariantLayer(
    #         input_dim=self.input_dim,
    #         input_channels=self.input_channels,
    #         output_channels=self.output_channels,
    #     )
    #
    #     batch = torch.rand([batch_size, self.input_channels, self.input_dim])
    #     results = layer(batch)
    #     perms = list(permutations(range(self.input_dim)))
    #     for perm in perms:
    #         assert np.all(
    #             (layer(batch[:, :, perm]) == results[:, :, perm]).numpy
    #         )
