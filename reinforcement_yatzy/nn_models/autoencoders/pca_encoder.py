import pandas as pd
from sklearn.decomposition import PCA
import torch

from reinforcement_yatzy.nn_models.autoencoders.general_encoder import GeneralEncoder
from reinforcement_yatzy.yatzy import ABCYatzyPlayer


class PCAEncoder(GeneralEncoder):
    def __init__(
        self,
        dataset: pd.DataFrame,  # TODO: add more options
        n_components: int,
    ):
        super().__init__()

        assert n_components <= ABCYatzyPlayer.NUM_ENTRIES, f'The number of components must be <= {ABCYatzyPlayer.NUM_ENTRIES}'
        self.latent_dim = n_components
        self.pca = PCA(n_components)
        self.pca.fit(dataset.values)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output = torch.tensor(self.pca.transform(batch), dtype=torch.float32)
        return output
