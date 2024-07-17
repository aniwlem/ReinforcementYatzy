from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any
import numpy as np


class Entries(IntEnum):
    ONES = 0
    TWOS = 1
    THREES = 2
    FOURS = 3
    FIVES = 4
    SIXES = 5
    ONE_PAIR = 6
    TWO_PAIRS = 7
    THREE_OF_A_KIND = 8
    FOUR_OF_A_KIND = 9
    SMALL_STRAIGHT = 10
    BIG_STRAIGHT = 11
    FULL_HOUSE = 12
    CHANCE = 13
    YATZY = 14


class ABCYatzyPlayer(ABC):
    UNPLAYED_VAL = -1
    SCRATCH_VAL = -2
    BONUS_VAL = 50
    # TODO: turn dict into enum and numpy array
    scoreboard = UNPLAYED_VAL * np.ones([len(Entries)], dtype=int)
    entry_names = [
        'Ones',
        'Twos',
        'Threes',
        'Fours',
        'Fives',
        'Sixes',
        'One Pair',
        'Two Pairs',
        'Three of a Kind',
        'Four of a Kind',
        'Small Straight',
        'Big Straight',
        'Full House',
        'Chance',
        'Yatzy',
    ]
    turns_left = 2
    curr_possible_scores = np.nan * np.ones_like(scoreboard)
    NUM_DICE = 5
    NUM_ENTRIES = scoreboard.size
    NUMS = [
        Entries.ONES,
        Entries.TWOS,
        Entries.THREES,
        Entries.FOURS,
        Entries.FIVES,
        Entries.SIXES
    ]
    bonus = UNPLAYED_VAL

    def __init__(self) -> None:
        self.dice = np.zeros(self.NUM_DICE, dtype=int)

    @abstractmethod
    def throw_dice(self, i_dice_throw) -> None:
        ...

    @abstractmethod
    def select_dice_to_throw(self) -> Any:
        ...

    @abstractmethod
    def select_next_entry(self) -> int:
        ...

    def check_score_current_dice(self) -> np.ndarray:
        '''
        Calculates the point for each scoreboard entry for the current dice.
        Illegal moves get a score of self.SCRATCH_VAL

        This implementation does NOT count four of a kind as two pairs, or
        yatzy as full house.
        '''

        # less magic
        die_sides = 6

        # dots is sorted, ascending
        raw_dots, raw_counts = np.unique(self.dice, return_counts=True)

        dots = np.arange(1, die_sides + 1)

        counts = np.zeros([die_sides])
        counts[raw_dots - 1] = raw_counts

        curr_possible_scores = self.SCRATCH_VAL * \
            np.ones([self.NUM_ENTRIES], dtype=int)

        curr_possible_scores[Entries.CHANCE] = np.sum(self.dice)

        # Upper section
        upper_scores = (dots * counts)
        upper_scores[upper_scores == 0] = self.SCRATCH_VAL
        curr_possible_scores[self.NUMS] = upper_scores

        # Lower section
        # tuplets, used later for score assignment
        n_tuples = 3  # 2, 3, 4 of a kind, yatzy is handled separetly

        n_tuple_scores = self.SCRATCH_VAL * \
            np.ones((die_sides, n_tuples), dtype=int)

        # each column is fixed value, [2, 3, 4]
        tuple_multipliers = np.tile(np.arange(2, 2 + n_tuples), (die_sides, 1))
        # each row is fixed value [1, 2, 3, 4, 5, 6].T
        dice_values = np.tile(np.arange(1, 1 + die_sides), (n_tuples, 1)).T

        # same as arr with "count >= i for i in range(2, 5)" stacked horizontally
        is_n_tuple = np.apply_along_axis(
            lambda arr: counts >= arr,
            0,
            tuple_multipliers
        )

        n_tuple_scores[is_n_tuple] = (
            tuple_multipliers * dice_values)[is_n_tuple]

        # One pair
        curr_possible_scores[Entries.ONE_PAIR] = int(
            np.max(n_tuple_scores[:, 0]))
        # Two pairs
        valid_pairs = n_tuple_scores[n_tuple_scores[:, 0] > 0, 0]
        if len(valid_pairs) > 1:
            # Since we only have 5 dice, if there are more than one pair there
            # are exactly thwo pairs
            curr_possible_scores[Entries.TWO_PAIRS] = int(
                np.sum(valid_pairs, dtype=int))
        # three of a kind
        curr_possible_scores[Entries.THREE_OF_A_KIND] = int(np.max(
            n_tuple_scores[:, 1]))
        # Four  of a kind
        curr_possible_scores[Entries.FOUR_OF_A_KIND] = int(np.max(
            n_tuple_scores[:, 2]))

        # Full House
        # if len(counts) == 2 the split is either 1:4 or 2:3
        if raw_counts.size == 2 and 2 in raw_counts:
            curr_possible_scores[Entries.FULL_HOUSE] = int(
                np.sum(self.dice, dtype=int))

        # Yatzy and straights
        # These are mutually exclusive so no need to check second if first is true
        if raw_counts.size == 1:
            print()
            curr_possible_scores[Entries.YATZY] = 50

        elif raw_counts.size == 5:
            if 1 in raw_dots and 6 not in raw_dots:
                curr_possible_scores[Entries.SMALL_STRAIGHT] = 15
            if 6 in raw_dots and 1 not in raw_dots:
                curr_possible_scores[Entries.BIG_STRAIGHT] = 20

        # Needs to be unable to choose things already played
        curr_possible_scores[
            self.scoreboard != self.UNPLAYED_VAL
        ] = self.SCRATCH_VAL

        self.curr_possible_scores = curr_possible_scores
        return curr_possible_scores

    @abstractmethod
    def play_turn(self):
        ...

    def reset_scoreboard(self):
        self.scoreboard = self.UNPLAYED_VAL * \
            np.ones_like(self.scoreboard, dtype=int)

    def play_game(self):
        self.reset_scoreboard()
        for _ in range(self.NUM_ENTRIES):
            self.play_turn()

    def check_bonus(self) -> int:
        upper_score, upper_is_full = self.get_upper_score()
        if upper_score >= 63:  # This number IS magical
            self.bonus = self.BONUS_VAL
        elif upper_is_full:
            self.bonus = self.SCRATCH_VAL
        return self.bonus

    def get_upper_score(self) -> tuple[int, bool]:
        upper_score = np.sum(
            self.scoreboard[
                (self.scoreboard != self.SCRATCH_VAL) *
                (self.scoreboard != self.UNPLAYED_VAL)
            ]
        )
        upper_is_full = self.UNPLAYED_VAL not in self.scoreboard[self.NUMS]
        return upper_score, upper_is_full

    # TODO: this gives non-scratching scores? i.e. this does not include entries
    # that one can scratch? YES
    def get_curr_legal_options(self) -> np.ndarray:
        self.curr_legal_options = (
            self.curr_possible_scores != self.SCRATCH_VAL)
        return self.curr_legal_options

    def get_scratch_options(self) -> np.ndarray:
        self.scratch_options = (self.scoreboard == self.UNPLAYED_VAL) *\
            (self.curr_possible_scores == self.SCRATCH_VAL)
        return self.scratch_options

    def get_total_score(self) -> int:
        bonus = (self.bonus if self.bonus == self.BONUS_VAL else 0)
        sum_entry_scores = np.sum(
            self.scoreboard[self.scoreboard != self.SCRATCH_VAL]
        )

        return sum_entry_scores + bonus
