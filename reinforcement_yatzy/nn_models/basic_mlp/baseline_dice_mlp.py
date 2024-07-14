import torch
from torch import nn


class BaseLineDiceMLP(nn.Module):
    def __init__(
        self,
        n_dice: int,
        n_entries: int,
        mlp_dims: list[int],
    ) -> None:
        super().__init__()

        mlp_layers = ([
            nn.Linear(
                n_dice + n_entries + 1,  # +1 for throw index
                mlp_dims[0],
            ),
            nn.ReLU(),
        ])

        # actual mlp
        for i in range(len(mlp_dims) - 1):
            mlp_layers.extend([
                nn.Linear(mlp_dims[i], mlp_dims[i + 1]),
                nn.ReLU(),
            ])

        mlp_layers.append(nn.Linear(mlp_dims[-1], n_dice))

        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(
        self,
        dices: torch.Tensor,
        scoreboards: torch.Tensor,
        throws_lefts: torch.Tensor
    ) -> torch.Tensor:
        batch = torch.concat([
            dices,
            scoreboards,
            throws_lefts.unsqueeze(1)  # turns into [batch_size, 1]
        ], dim=1)

        for layer in self.mlp_layers:
            batch = layer(batch)

        return batch
