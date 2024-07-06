import os
from pathlib import Path

import numpy as np
import pandas as pd

from reinforcement_yatzy.yatzy.empty_training_player import TrainingYatzyPlayer


class ScoreboardGenerator:
    '''
    Plays a sequence of games, saving each intermediate scoreboard in the games
    to a buffer. Saves the game buffer to csv.
    '''

    def __init__(
        self,
        save_path: Path,
    ) -> None:
        self.player = TrainingYatzyPlayer()
        self.save_path = save_path

        self.FULL_HOUSE_SCORES = np.array(list(set([
            2 * i + 3 * j for i, j in zip(range(1, 7), range(1, 7))
        ])))

        self.NUM_HOUSE_SCORES = len(self.FULL_HOUSE_SCORES)
        headers = list(self.player.scoreboard.keys())

        if not os.path.exists(save_path):
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.save_path, index=False)

    def generate_scoreboards(
        self,
        batch_size: int,
        n_unplayed: int,
        n_scratch: int
    ):
        '''
        Generates batch_size random valid yatzy scoreboards with n_unplayed 
        randomly selected entries with UNPLAYED_VAL and n_scratch randomly 
        selected entries with SCRATCH_VAL.
        '''

        '''
        A scoreboard has the following number of options:
        Upper section: 5 for each entry

        pair: 6
        two pair: 6 * 5
        Three of a Kind: 6
        Four of a Kind: 6

        Small Straight: 1
        Big Straight: 1

        Full House: 6 * 5
        Chance: 30
        Yatzy: 1

        Total number of boards:
        (6 ** 5) * 6 * (6 * 5) * 6 * 6 * 1 * 1 * 6 * 5 * 30 * 1 \approx 10 ** 10

        Would take up >> 10 GB of disk space, can't go through all of them.
        Sample the scoreboards randomly.
        '''

        assert n_unplayed + n_scratch <= self.player.NUM_ENTRIES, 'n_unplayed + n_scratch must be less than the scoreboard size'

        # Only create the valid options, the UNPLAYED_VAL and SCRATCH_VAL can be added afterwards.
        upper_randoms = np.tile(np.arange(1, 7), [batch_size, 1]) * \
            np.random.randint(1, 7, [batch_size, 6])
        pair_randoms = 2 * np.random.randint(1, 7, [batch_size, 1])
        # scores are between 6 and 22, all even numbers possible
        two_pair_randoms = 2 * np.random.randint(3, 12, [batch_size, 1])
        three_of_a_kind_randoms = 3 * np.random.randint(1, 7, [batch_size, 1])
        four_of_a_kind_randoms = 3 * np.random.randint(1, 7, [batch_size, 1])
        small_straight_randoms = np.tile([15], [batch_size, 1])
        big_straight_randoms = np.tile([20], [batch_size, 1])
        # The possible scores for full house in non-contiguous, easiest way to
        # select valid ones is to sample the valid numbers
        full_house_randoms = self.FULL_HOUSE_SCORES[
            np.random.randint(
                self.NUM_HOUSE_SCORES,
                size=[batch_size, 1]
            )
        ]
        chance = np.random.randint(1, 30, [batch_size, 1])
        yatzy_random = np.tile([50], [batch_size, 1])

        scoreboards = np.concat([
            upper_randoms,
            pair_randoms,
            two_pair_randoms,
            three_of_a_kind_randoms,
            four_of_a_kind_randoms,
            small_straight_randoms,
            big_straight_randoms,
            full_house_randoms,
            chance,
            yatzy_random,
        ],
            axis=1
        )

        # TODO: add the bonus in somewhere

        indices = np.arange(self.player.NUM_ENTRIES)

        unplayed_inds = np.array([
            np.random.choice(indices, n_unplayed, replace=False)
            for _ in range(batch_size)
        ])

        scratch_inds = np.array([
            np.random.choice(indices, n_scratch, replace=False)
            for _ in range(batch_size)
        ])

        scoreboards[unplayed_inds] = self.player.UNPLAYED_VAL
        scoreboards[scratch_inds] = self.player.SCRATCH_VAL

        return pd.DataFrame(scoreboards)

    def append_chunks(self, batch_size: int, n_chunks: int):
        for i in range(n_chunks):
            n_unplayed = np.random.randint(self.player.NUM_ENTRIES)
            n_scratch = np.random.randint(self.player.NUM_ENTRIES - n_unplayed)
            new_data = self.generate_scoreboards(
                batch_size,
                n_unplayed,
                n_scratch
            )

            new_data.to_csv(
                self.save_path,
                mode='a',
                header=False,
                index=False,
            )
