import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from reinforcement_yatzy.scoreboard_dataset.scoreboard_dataset import ScoreboardDataset
from reinforcement_yatzy.nn_models.autoencoders.scoreboard_autoencoder import ScoreboardEncoder, ScoreboardDecoder, ScoreboardAutoencoder
from reinforcement_yatzy.yatzy.empty_training_player import TrainingYatzyPlayer


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
        with tqdm(data_loader, unit='batch') as tepoch:
            for batch in tepoch:
                tepoch.set_description(f'Epoch {epoch}')

                optimizer.zero_grad()
                outputs = autoencoder(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
    return autoencoder


def main(save_path: Path, dataset_path: Path, epochs: int):
    encoder_dims = [32, 16, 8]
    latent_dim = 4
    autoencoder = setup_autoencoder(encoder_dims, latent_dim)

    dataset_df = pd.read_csv(dataset_path)
    dataset = ScoreboardDataset(dataset_df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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
    parser.add_argument("dataset_path", type=Path,
                        help="Path to the scoreboard dataset")
    parser.add_argument("epochs", type=int,
                        help="Number of epochs to train")
    args = parser.parse_args()

    main(**vars(args))
