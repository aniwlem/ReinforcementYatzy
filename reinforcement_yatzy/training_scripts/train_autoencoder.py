import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from ..nn_models.autoencoders.scoreboard_autoencoder import ScoreboardEncoder, ScoreboardDecoder, ScoreboardAutoencoder
# from reinforcement_yatzy.nn_models.autoencoders.scoreboard_autoencoder
from ..yatzy.empty_training_player import TrainingYatzyPlayer
# from reinforcement_yatzy.yatzy.empty_training_player import TrainingYatzyPlayer


class ScoreboardDataset(Dataset):
    def __init__(self, buffer_size: int, p_replace: float = 0) -> None:
        super().__init__()
        self.player = TrainingYatzyPlayer()
        self.buffer_size = buffer_size
        self.buffer = [self._get_scoreboard() for _ in range(self.buffer_size)]
        self.p_replace = p_replace

    def _get_scoreboard(self):
        self.player.throw_dice(range(self.player.NUM_DICE))
        self.player.check_points_of_dice()

        return torch.Tensor(list(self.player.curr_possible_scores.values()))

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, index) -> torch.Tensor:
        if np.random.rand() < self.p_replace:
            self.buffer[index] = self._get_scoreboard()
        return self.buffer[index]


def setup_autoencoder(encoder_dims: list[int], latent_dim: int) -> ScoreboardAutoencoder:
    input_dim = TrainingYatzyPlayer.NUM_ENTRIES

    encoder = ScoreboardEncoder(
        input_dim=input_dim,
        mlp_dims=encoder_dims,
        latent_dim=latent_dim
    )

    decoder = ScoreboardDecoder(
        input_dim=input_dim,
        mlp_dims=encoder_dims[::-1],
        latent_dim=latent_dim
    )

    autoencoder = ScoreboardAutoencoder(
        encoder=encoder,
        decoder=decoder,
    )
    return autoencoder


def train_autoencoder(
    autoencoder: ScoreboardAutoencoder,
    data_loader: DataLoader,
    epochs: int,
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for batch in data_loader:
            outputs = autoencoder(batch)
            optimizer.zero_grad()
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch} loss: {loss:.3e}')
    return autoencoder


def main(save_path: Path):
    encoder_dims = [32, 16, 8]
    latent_dim = 4
    autoencoder = setup_autoencoder(encoder_dims, latent_dim)

    dataset = ScoreboardDataset(buffer_size=1024, p_replace=0.1)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 1001
    train_autoencoder(autoencoder, dataloader, epochs)
    torch.save(autoencoder.encoder.state_dict(), save_path)
    print(f'Saved weights to {save_path}')

    player = TrainingYatzyPlayer()
    player.throw_dice(range(5))
    player.check_points_of_dice()
    in_board = torch.Tensor(list(player.curr_possible_scores.values()))
    out_board = autoencoder(in_board)
    print(
        f'input: {in_board.detach().numpy()}\noutput: {out_board.detach().numpy()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a neural network and save the weights.")
    parser.add_argument("save_path", type=Path,
                        help="Path to save the model weights.")
    args = parser.parse_args()

    main(args.save_path)
