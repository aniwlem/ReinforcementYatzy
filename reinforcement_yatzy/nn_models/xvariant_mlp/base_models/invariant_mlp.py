from torch import nn

from reinforcement_yatzy.nn_models.xvariant_mlp.base_models.equivariant_mlp import EquivariantMLP
from reinforcement_yatzy.nn_models.xvariant_mlp.pool_type_enum import PoolType

from dataclasses import dataclass
import torch


@dataclass
class InvariantPoolingParams:
    embed_pooling: PoolType
    elem_pooling: PoolType


class InvariantPooling(nn.Module):
    '''
    Class that applies three successive pooling layers to the output of a
    permutation equivariant layer with multiple channels and a set of vectorial
    embeddings.
    '''

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        n_channels: int,
        embed_pooling: PoolType,
        elem_pooling: PoolType
    ) -> None:
        super().__init__()

        self.embed_pooling = embed_pooling.layer(embed_dim)
        self.elem_pooling = elem_pooling.layer(seq_len)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        '''
        inputs: shape[batch_size, input_channels, n_elems, embed_dim]

        outputs: shape[batch_size]

        Pooling is done in order of atomicity: embeddings, then elements
        '''

        if len(batch.shape) == 3:
            batch.unsqueeze(0)

        batch_size, n_channels, n_dice, embed_dim = batch.shape
        # Pool away embedding dimension
        embed_batch = batch.reshape(
            [batch_size * n_channels, n_dice, embed_dim])
        embed_pooled = self.embed_pooling(embed_batch).reshape([
            batch_size, n_channels, n_dice
        ])

        # Pool away element dimension
        elem_pooled = self.elem_pooling(embed_pooled).squeeze(-1)

        return elem_pooled


class InvariantMLP(EquivariantMLP):
    '''
    Permutation invariant multilayer NN for working on sets of vectors. For any
    permutation pi(X) of the input X, the output Y is guaranteed to satisfy
    Y(X) = Y(pi(X))

    The model is in essense a equivariant model with added pooling layers at the
    end. A single invariant layer is uninsteresting since any following layers 
    will preserve invariance, and thus could be anything, and a single layer 
    most likely isn't enoug for a meaningful transformation.
    '''

    def __init__(
        self,
        n_elems: int,
        embed_dim: int,
        mlp_channels: list[int],
        mlp_pool_type: PoolType,
        invarintifier_pool_types: InvariantPoolingParams,
        # num_dice: int,
        # num_entries: int,
        # num_poolings: int,
        # invariant_hyperparam: int,
        # mlp_dims: list[int],
    ):
        # The super class will give an equivariant layer, to make an invariant
        # layer one just has to pool away all dimensions except batch.
        super().__init__(
            n_elems=n_elems,
            embed_dim=embed_dim,
            mlp_channels=mlp_channels,
            pool_type=mlp_pool_type,
        )

        self.invarintifier = InvariantPooling(
            embed_dim=embed_dim,
            seq_len=n_elems,
            n_channels=mlp_channels[-1],
            embed_pooling=invarintifier_pool_types.embed_pooling,
            elem_pooling=invarintifier_pool_types.elem_pooling,
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        '''
        inputs: shape[batch_size, input_channels, n_elems, embed_dim]

        outputs: shape[batch_size, output_channels, n_elems, embed_dim]
        '''
        equivariant_output = super().forward(batch)

        # single batch
        if len(equivariant_output.shape) == 3:
            equivariant_output.unsqueeze(0)

        invariant_output = self.invarintifier(equivariant_output)
        print(invariant_output.shape)
        return invariant_output
