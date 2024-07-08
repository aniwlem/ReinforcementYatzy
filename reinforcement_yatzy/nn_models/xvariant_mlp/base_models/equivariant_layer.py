import torch
from torch import nn

from reinforcement_yatzy.nn_models.xvariant_mlp.pool_type_enum import PoolType


class EquivariantLayer(nn.Module):
    '''
    Permutation equivariant NN-layer for working on sets of vectors. For any
    permutation pi(X) of the input X, the output Y is guaranteed to satisfy
    pi(Y(X)) = Y(pi(X))
    '''

    def __init__(
        self,
        n_elems: int,
        embed_dim: int,
        n_input_channels: int,
        n_output_channels: int,
        pool_type: PoolType,
    ) -> None:
        super().__init__()

        self.n_elems = n_elems
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.embed_dim = embed_dim
        self.lambda_ = nn.Parameter(torch.rand([
            n_input_channels, n_output_channels
        ]))

        self.gamma = nn.Parameter(torch.rand([
            n_input_channels, n_output_channels
        ]))

        self.pool_func = pool_type.layer(n_elems)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        '''
        inputs: shape[batch_size, input_channels, n_elems, embed_dim]

        outputs: shape[batch_size, output_channels, n_elems, embed_dim]
        '''
        # need batch dim
        if len(batch.shape) == 3:
            batch.unsqueeze(0)

        batch_size, input_channels, n_elems, embed_dim = batch.shape

        # To multiply 1:st dim with matrix
        prop_reshape_batch = batch.permute([0, 2, 3, 1])
        prop_term = (prop_reshape_batch @ self.lambda_).permute([0, 3, 1, 2])

        # Want to pool over the set elements, permute to get them last, reshape
        # to get correct number of dims for 1dpooling
        common_reshape_batch = batch.permute([0, 1, 3, 2]).reshape([
            batch_size * input_channels * embed_dim, n_elems
        ])
        common_funced = self.pool_func(common_reshape_batch)
        # Undo reshape/permutation. Because a permutation is needed for the next
        # multiplication with a matrix, the permutation here is not the inverse
        # of the one before.
        common_funced = common_funced.reshape([
            batch_size, input_channels, 1, embed_dim
        ]).permute([0, 3, 2, 1])

        common_term = (common_funced @ self.gamma).permute([0, 3, 2, 1])

        return prop_term + common_term
