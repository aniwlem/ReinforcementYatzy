from pathlib import Path

import torch
from torch import nn

from reinforcement_yatzy.nn_models.autoencoders.general_encoder import GeneralEncoder


class MLPScoreboardEncoder(GeneralEncoder):
    def __init__(
        self,
        n_entries: int,
        latent_dim: int,
        mlp_dims: list[int],
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        mlp_layers = ([
            nn.Linear(
                n_entries,
                mlp_dims[0],
            ),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
        ])

        # actual mlp
        for i in range(len(mlp_dims) - 1):
            mlp_layers.extend([
                nn.Linear(mlp_dims[i], mlp_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            ])

        mlp_layers.append(nn.Linear(mlp_dims[-1], latent_dim))

        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp_layers:
            batch = layer(batch)
        return batch


class MLPScoreboardDecoder(nn.Module):
    def __init__(
        self,
        n_entries: int,
        latent_dim: int,
        mlp_dims: list[int],
        p_dropout: float,
    ) -> None:
        super().__init__()

        mlp_layers = ([
            nn.Linear(
                latent_dim,
                mlp_dims[0],
            ),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
        ])

        for i in range(len(mlp_dims) - 1):
            mlp_layers.extend([
                nn.Linear(mlp_dims[i], mlp_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),
            ])

        mlp_layers.append(nn.Linear(mlp_dims[-1], n_entries))

        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp_layers:
            batch = layer(batch)
        return batch


class MLPScoreboardAutoencoder(nn.Module):
    def __init__(
        self,
        n_entries: int,
        latent_dim: int,
        mlp_dims: list[int],
        p_dropout: float,
        mlp_dims_decoder: list[int] | None = None,
    ) -> None:
        super().__init__()
        if mlp_dims_decoder is None:
            mlp_dims_decoder = mlp_dims[::-1]

        self.encoder = MLPScoreboardEncoder(
            n_entries,
            latent_dim,
            mlp_dims,
            p_dropout,
        )

        self.decoder = MLPScoreboardDecoder(
            n_entries,
            latent_dim,
            mlp_dims_decoder,
            p_dropout,
        )

    def load_encoder_decoder_state_dicts(
        self,
        encoder_state_dict_path: Path,
        decoder_state_dict_path: Path,
    ) -> None:
        self.encoder.load_state_dict(torch.load(encoder_state_dict_path))
        self.decoder.load_state_dict(torch.load(decoder_state_dict_path))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(batch)
        output = self.decoder(latent)
        return output
