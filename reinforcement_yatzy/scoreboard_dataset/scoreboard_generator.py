import os
from pathlib import Path

import numpy as np
import pandas as pd

from reinforcement_yatzy.yatzy.base_player import ABCYatzyPlayer


class ScoreboardGenerator:
    '''
    Generates a dataset of valid yatzy scoreboards, either according to some 
    constraints w.r.t. number of unplayed values and scratched values, or following
    the distribution of a randomly played game.
    '''

    def __init__(
        self,
        save_path: Path | None = None,
    ) -> None:
        self.save_path = save_path

        self.FULL_HOUSE_SCORES = np.array([
            2 * i + 3 * j
            for i in range(1, 7) for j in range(1, i)
        ])

        self.NUM_HOUSE_SCORES = len(self.FULL_HOUSE_SCORES)

        # When selecting which entries to unplay or scratch, do it proportional
        # to the to the number of possible combinations for that entry. All numbers
        # were calculated by brute force, to assure no combinatorial mistakes
        # were made
        #
        # Ones - Sixes: 4651 for every entry
        #
        # One Pair: 1526 * 6 = 9165
        #
        # Two Pairs: 140 * 6 * 5 = 4200
        #
        # Three of a Kind: 276 * 6 = 1656
        #
        # Four of a kind: 26 * 6 = 156
        #
        # Small Straight: 120
        #
        # Big Straight: 120
        #
        # Full House: 10 * 6 * 5 = 300
        #
        # Chance: 6 ** 5 = 7776
        #
        # Yatzy: 1 * 6

        self.probs = np.array([
            4651, 4651, 4651, 4651, 4651, 4651,
            9165,
            4200,
            1656,
            156,
            120,
            120,
            300,
            7776,
            6
        ])
        # make the tails less fat
        self.probs = np.sqrt(self.probs)
        self.probs /= np.sum(self.probs)

        headers = ABCYatzyPlayer.entry_names

        if save_path is not None:
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
        A filled in scoreboard has the following number of options:
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

        Would take up >> 10 GB of disk space, not including partially filled or
        with scratched values. Way too big to go through all of them, sample the
        scoreboards randomly.
        '''

        assert n_unplayed + n_scratch <= ABCYatzyPlayer.NUM_ENTRIES, 'n_unplayed + n_scratch must be less than the scoreboard size'

        # Only create the valid options, the UNPLAYED_VAL and SCRATCH_VAL can be added afterwards.
        upper_randoms = np.tile(np.arange(1, 7), [batch_size, 1]) * \
            np.random.randint(1, 6, [batch_size, 6])
        pair_randoms = 2 * np.random.randint(1, 7, [batch_size, 1])
        # scores are between 6 and 22, all even numbers possible
        two_pair_randoms = 2 * np.random.randint(3, 12, [batch_size, 1])
        three_of_a_kind_randoms = 3 * np.random.randint(1, 7, [batch_size, 1])
        four_of_a_kind_randoms = 4 * np.random.randint(1, 7, [batch_size, 1])
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

        scoreboards = np.concat(
            [
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

        n_after_unplayed = ABCYatzyPlayer.NUM_ENTRIES - n_unplayed
        single_unplayed_mask = np.concat([
            np.ones([n_unplayed], dtype=bool),
            np.zeros([n_after_unplayed], dtype=bool)
        ])

        unplayed_mask = np.array([
            np.random.permutation(single_unplayed_mask)
            for _ in range(batch_size)
        ])

        # Only scratch played values, to guarantee #unplayed == n_unplayed
        # on each row
        single_scratch_mask = np.concat([
            np.ones([n_scratch], dtype=bool),
            np.zeros([n_after_unplayed - n_scratch], dtype=bool)
        ])

        # mask for the played part of a scoreboard
        proto_scratch_mask = np.concat([
            np.random.permutation(single_scratch_mask)
            for _ in range(batch_size)
        ], axis=0)

        scratch_mask = np.zeros_like(unplayed_mask).astype(bool)
        scratch_mask[~unplayed_mask] = proto_scratch_mask

        scoreboards[unplayed_mask] = ABCYatzyPlayer.UNPLAYED_VAL
        scoreboards[scratch_mask] = ABCYatzyPlayer.SCRATCH_VAL

        return pd.DataFrame(scoreboards).astype('int8')

    def append_chunks(self, batch_size: int, n_batches: int):
        for _ in range(n_batches):
            n_unplayed = np.random.randint(ABCYatzyPlayer.NUM_ENTRIES)
            n_scratch = np.random.randint(
                ABCYatzyPlayer.NUM_ENTRIES - n_unplayed
            )

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
