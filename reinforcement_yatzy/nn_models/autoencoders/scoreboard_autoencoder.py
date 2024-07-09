from pathlib import Path

import torch
from torch import nn


class ScoreboardEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        latent_dim: int = 8,
        mlp_dims: list[int] = [69],
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        mlp_layers = ([
            nn.Linear(
                input_dim,
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

        mlp_layers.append(nn.Linear(mlp_dims[-1], latent_dim))

        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp_layers:
            batch = layer(batch)
        return batch


class ScoreboardDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        latent_dim: int = 8,
        mlp_dims: list[int] = [69],
    ) -> None:
        super().__init__()

        mlp_layers = ([
            nn.Linear(
                latent_dim,
                mlp_dims[0],
            ),
            nn.ReLU(),
        ])

        for i in range(len(mlp_dims) - 1):
            mlp_layers.extend([
                nn.Linear(mlp_dims[i], mlp_dims[i + 1]),
                nn.ReLU(),
            ])

        mlp_layers.append(nn.Linear(mlp_dims[-1], input_dim))

        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp_layers:
            batch = layer(batch)
        return batch


class ScoreboardAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        latent_dim: int = 8,
        mlp_dims: list[int] = [69],
        mlp_dims_decoder: list[int] | None = None,
    ) -> None:
        super().__init__()
        if mlp_dims_decoder is None:
            mlp_dims_decoder = mlp_dims[::-1]

        self.encoder = ScoreboardEncoder(
            input_dim,
            latent_dim,
            mlp_dims,
        )

        self.decoder = ScoreboardDecoder(
            input_dim,
            latent_dim,
            mlp_dims_decoder,
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
