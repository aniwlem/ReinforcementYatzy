import pandas as pd
import torch
from torch.utils.data import Dataset


class ScoreboardDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        super().__init__()
        self.df = dataframe

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index) -> torch.Tensor:
        curr_entry = self.df.iloc[index, :]
        return torch.Tensor(curr_entry.values)
