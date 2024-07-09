import pandas as pd
import torch
from torch.utils.data import ChunkDataReader

ChunkDataReader


class ScoreboardDataset(Dataset):
    # TODO: instead of loading up the entire csv, take chunks of it, and iterate
    # over chunks of. SO perhaps for the first n batches all entries will come from the
    # m:th chunk, when all is done in chunk m, chunk p will be samlped from etc.
    def __init__(self, dataframe: pd.DataFrame) -> None:
        super().__init__()
        self.df = dataframe

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index) -> torch.Tensor:
        curr_entry = self.df.iloc[index, :]
        return torch.Tensor(curr_entry.values)
