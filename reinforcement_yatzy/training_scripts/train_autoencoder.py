import argparse
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
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
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    loss_path: Path,
    n_rising_until_break: int,
):
    '''Train the autoencoder with holdout validation'''
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    n_epochs_val_rising = 0
    train_losses = np.zeros([n_epochs])
    val_losses = np.zeros([n_epochs])

    for i_epoch in range(n_epochs):
        # Training
        with tqdm(train_loader, unit='batch') as tepoch:
            sum_loss = 0
            for i_batch, batch in enumerate(tepoch):
                tepoch.set_description(f'Epoch {i_epoch}')

                optimizer.zero_grad()
                outputs = autoencoder(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()

                tepoch.set_postfix(loss=sum_loss / (i_batch + 1))

        avg_train_loss = sum_loss / len(train_loader)
        train_losses[i_epoch] = avg_train_loss

        # Validation
        with torch.no_grad():
            sum_val_loss = 0
            for batch in val_loader:
                outputs = autoencoder(batch)
                val_loss = criterion(outputs, batch)
                sum_val_loss += val_loss

        avg_val_loss = sum_val_loss / len(val_loader)
        val_losses[i_epoch] = avg_val_loss

        tqdm.write(
            f'\nEpoch {i_epoch} - Training Loss: {avg_train_loss:.2e} - Validation Loss: {avg_val_loss:.2e}\n')

        # Holdout validation
        if i_epoch > 0 and val_losses[i_epoch] > val_losses[i_epoch - 1]:
            n_epochs_val_rising += 1

        if n_epochs_val_rising == n_rising_until_break:
            break

    np.savetxt(
        loss_path,
        np.column_stack([train_losses, val_losses]),
        delimiter=',',
        header='train_loss,val_loss',
    )

    return autoencoder


def main(save_path: Path, dataset_path: Path, loss_log_dir_path: Path, epochs: int):
    assert os.path.exists(loss_log_dir_path), 'Invalid directory for loss log'

    encoder_dims = [32, 32, 16, 16, 8]
    latent_dim = 6
    autoencoder = setup_autoencoder(encoder_dims, latent_dim)

    dataset_df = pd.read_csv(dataset_path)
    dataset = ScoreboardDataset(dataset_df)
    train_set, val_set = random_split(dataset, [.95, 0.05])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

    now = datetime.now()
    formatted_time = now.strftime("%m-%d_%H-%M")
    loss_file_path = Path(''.join([
        'auto_encoder_',
        '[',
        *[str(dim) + '_' for dim in encoder_dims],
        ']_',
        f'{latent_dim}_',
        formatted_time,
        '.csv'
    ]))
    loss_path = Path(os.path.join(loss_log_dir_path, loss_file_path))
    train_autoencoder(
        autoencoder,
        train_loader,
        val_loader,
        epochs,
        loss_path=loss_path,
        n_rising_until_break=5
    )
    torch.save(autoencoder.encoder.state_dict(), save_path)
    print(f'Saved weights to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a neural network and save the weights.")
    parser.add_argument("save_path", type=Path,
                        help="Path to save the model weights.")
    parser.add_argument("dataset_path", type=Path,
                        help="Path to the scoreboard dataset")
    parser.add_argument("loss_log_dir_path", type=Path,
                        help="Path to the directory with loss logs")
    parser.add_argument("epochs", type=int,
                        help="Number of epochs to train")
    args = parser.parse_args()

    main(**vars(args))
