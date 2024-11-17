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

    @classmethod
    def normalize_and_embed(cls, batch: torch.Tensor) -> torch.Tensor:
        '''
        Function that converts the scalar values, with sentinel values, in a 
        scoreboard, to a (per entry) normalized vectorial representation:
        [flag, value]
        where [0, value] means the value should be read as a normal value and
        [1, flag_type] means the value is either unplayed (flag_type == 0) or 
        scratched (flag_type == 1)

        Instead of adding a new dimension for this extra value they are 
        prepended before each value. So a scoreboard of shape [NUM_ENTRIES] 
        gets an embedding of shape [2 * NUM_ENTRIES]

        inputs:
        batch: [batch_size, NUM_ENTRIES]

        output: [batch_size, NUM_ENTRIES * 2]
        '''

        if len(batch.shape) == 1:
            batch = batch.unsqueeze(0)

        batch_size = batch.shape[0]

        is_unplayed = batch == ABCYatzyPlayer.UNPLAYED_VAL
        is_scratched = batch == ABCYatzyPlayer.SCRATCH_VAL
        is_special = is_unplayed | is_scratched

        output = torch.zeros([batch_size, 2 * ABCYatzyPlayer.NUM_ENTRIES])

        reshaped_norm_table = cls.normalization_table.unsqueeze(0)\
            .expand([batch_size, -1])
        output[:, 1::2][~is_special] = batch[~is_special] / \
            reshaped_norm_table[~is_special]

        output[:, ::2] = is_special
        output[:, 1::2][is_unplayed] = cls.UNPLAYED_FLAG
        output[:, 1::2][is_scratched] = cls.SCRATCH_FLAG

        return output

    @classmethod
    def undo_normalize_and_embed(cls, batch: torch.Tensor) -> torch.Tensor:
        '''
        Function that converts the encoded, normalized representation of the 
        scoreboard, back to the scalar values.

        inputs:
        batch: [batch_size, NUM_ENTRIES * 2]

        output: [batch_size, NUM_ENTRIES]
        '''

        if len(batch.shape) == 1:
            batch = batch.unsqueeze(0)

        batch_size, num_entries_x_2 = batch.shape
        num_entries = num_entries_x_2 // 2

        output = torch.zeros([batch_size, num_entries])

        is_special = batch[:, ::2] == 1
        # Must only check special part otherwise maxed scores will be part of
        # scratch mask
        is_unplayed = batch[:, 1::2][is_special] == cls.UNPLAYED_FLAG
        is_scratched = batch[:, 1::2][is_special] == cls.SCRATCH_FLAG

        reshaped_norm_table = cls.normalization_table.unsqueeze(0)\
            .expand([batch_size, -1])

        output[~is_special] = batch[:, 1::2][~is_special] * \
            reshaped_norm_table[~is_special]

        output[is_special][is_unplayed] = ABCYatzyPlayer.UNPLAYED_VAL
        output[is_special][is_scratched] = ABCYatzyPlayer.SCRATCH_VAL

        return output
