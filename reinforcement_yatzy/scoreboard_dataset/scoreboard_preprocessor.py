import torch

from reinforcement_yatzy.yatzy.base_player import ABCYatzyPlayer


class ScoreboardPreProcessor:
    normalization_table = torch.tensor([
        5, 10, 15, 20, 25, 30,  # Upper section
        12, 22, 18, 24,  # n pairs, tuples
        15, 20,  # straights
        28, 30, 50  # five dice "hands"
    ],
        dtype=torch.float32  # So there's no need to convert from int
    )
    UNPLAYED_FLAG = 0
    SCRATCH_FLAG = 1

    @classmethod
    def normalize(cls, batch: torch.Tensor) -> torch.Tensor:
        '''
        Function that normalizes non-sentinel values in a scoreboard 
        (per entry), while keeping the sentinel values.

        inputs:
        batch: [batch_size, NUM_ENTRIES]

        output: [batch_size, NUM_ENTRIES]
        '''
        if len(batch.shape) == 1:
            batch = batch.unsqueeze(0)

        batch_size = batch.shape[0]
        is_unplayed = batch == ABCYatzyPlayer.UNPLAYED_VAL
        is_scratched = batch == ABCYatzyPlayer.SCRATCH_VAL
        is_special = is_unplayed | is_scratched

        reshaped_norm_table = cls.normalization_table.unsqueeze(
            0).expand([batch_size, -1])
        batch[~is_special] = batch[~is_special] / \
            reshaped_norm_table[~is_special]

        return batch

    @classmethod
    def undo_normalize(cls, batch: torch.Tensor) -> torch.Tensor:
        '''
        Function that un-normalizes non-sentinel values in a scoreboard 
        (per entry), while keeping the sentinel values.

        inputs:
        batch: [batch_size, NUM_ENTRIES]

        output: [batch_size, NUM_ENTRIES]
        '''
        if len(batch.shape) == 1:
            batch = batch.unsqueeze(0)

        batch_size = batch.shape[0]
        is_unplayed = batch == ABCYatzyPlayer.UNPLAYED_VAL
        is_scratched = batch == ABCYatzyPlayer.SCRATCH_VAL
        is_special = is_unplayed | is_scratched

        reshaped_norm_table = cls.normalization_table.unsqueeze(
            0).expand([batch_size, -1])
        batch[~is_special] = batch[~is_special] * \
            reshaped_norm_table[~is_special]

        return batch
