import pytest

import torch

from reinforcement_yatzy.nn_models.autoencoders.scoreboard_autoencoder import ScoreboardEncoder, ScoreboardAutoencoder


class TestAutoEncoder:
    input_dim = 2
    latent_dim = 3
    mlp_dims = [5, 7, 11]

    @pytest.fixture
    def encoder(self):
        encoder = ScoreboardEncoder(
            self.input_dim,
            self.latent_dim,
            self.mlp_dims,
        )
        return encoder

    @pytest.fixture
    def autoencoder(self):
        autoencoder = ScoreboardAutoencoder(
            self.input_dim,
            self.latent_dim,
            self.mlp_dims,
        )
        return autoencoder

    @pytest.mark.parametrize('batch_size', range(10))
    def test_encoder(self, batch_size: int, encoder: ScoreboardEncoder):
        batch = torch.rand([batch_size, self.input_dim])
        assert list(encoder(batch).shape) == [batch_size, self.latent_dim]

    @pytest.mark.parametrize('batch_size', range(10))
    def test_autoencoder(self, batch_size: int, autoencoder: ScoreboardAutoencoder):
        batch = torch.rand([batch_size, self.input_dim])
        assert list(autoencoder(batch).shape) == [
            batch_size, self.input_dim]
