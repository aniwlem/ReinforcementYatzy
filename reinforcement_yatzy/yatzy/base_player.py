from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class ABCYatzyPlayer(ABC):
    UNPLAYED_VAL = -1
    SCRATCH_VAL = -2
    BONUS_VAL = 50
    scoreboard = {
        'Ones': UNPLAYED_VAL,
        'Twos': UNPLAYED_VAL,
        'Threes': UNPLAYED_VAL,
        'Fours': UNPLAYED_VAL,
        'Fives': UNPLAYED_VAL,
        'Sixes': UNPLAYED_VAL,
        'One Pair': UNPLAYED_VAL,
        'Two Pairs': UNPLAYED_VAL,
        'Three of a Kind': UNPLAYED_VAL,
        'Four of a Kind': UNPLAYED_VAL,
        'Small Straight': UNPLAYED_VAL,
        'Big Straight': UNPLAYED_VAL,
        'Full House': UNPLAYED_VAL,
        'Chance': UNPLAYED_VAL,
        'Yatzy': UNPLAYED_VAL,
    }

    turns_left = 2
    curr_possible_scores = {}
    NUMS = ['Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes']
    NUM_DICE = 5
    NUM_ENTRIES = len(scoreboard)
    bonus = UNPLAYED_VAL

    def __init__(self) -> None:
        self.dice = np.zeros(self.NUM_DICE, dtype=int)

    def throw_dice(
        self,
        ind_throw: list[int] | range | npt.NDArray[np.int_]
    ) -> None:
        new_vals = np.random.randint(1, 7, [len(ind_throw)])
        self.dice[ind_throw] = new_vals

    @abstractmethod
    def select_dice_to_throw(self) -> list[int]:
        ...

    @abstractmethod
    def select_next_entry(self) -> str:
        ...

    def check_score_current_dice(self) -> dict[str, int]:
        '''
        Calculates the point for each scoreboard entry for the current dice.
        Illegal moves get a score of self.SCRATCH_VAL
        '''
        # This implementation does NOT count four of a kind as two pairs, or
        # yatzy as full house.

        # less magic
        die_sides = 6

        # dots is sorted, ascending
        raw_dots, raw_counts = np.unique(self.dice, return_counts=True)

        dots = np.arange(1, die_sides + 1)

        counts = np.zeros((die_sides))
        counts[raw_dots - 1] = raw_counts

        curr_points = {key: self.SCRATCH_VAL for key in self.scoreboard.keys()}

        curr_points['Chance'] = np.sum(self.dice)

        # Upper section
        upper_scores = (dots * counts)
        upper_scores[upper_scores == 0] = self.SCRATCH_VAL
        for dot in dots:
            curr_points[self.NUMS[dot - 1]] = int(upper_scores[dot - 1])

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
        curr_points['One Pair'] = np.max(n_tuple_scores[:, 0]).astype(int)
        # Two pairs
        valid_pairs = n_tuple_scores[n_tuple_scores[:, 0] > 0, 0]
        if len(valid_pairs) > 1:
            # The slicing is not necessary since we only have 5 dice
            curr_points['Two Pairs'] = np.sum(valid_pairs[-2:], dtype=int)
        # three of a kind
        curr_points['Three of a Kind'] = np.max(
            n_tuple_scores[:, 1]).astype(int)
        # Four  of a kind
        curr_points['Four of a Kind'] = np.max(
            n_tuple_scores[:, 2]).astype(int)

        # Full House
        # if len(counts) == 2 the split is either 1:4 or 2:3
        if raw_counts.size == 2 and 2 in raw_counts:
            curr_points['Full House'] = np.sum(self.dice, dtype=int)

        # Yatzy and straights
        # These are mutually exclusive so no need to check second if first is true
        if raw_counts.size == 1:
            curr_points['Yatzy'] = 50

        elif raw_counts.size == 5:
            if 1 in raw_dots and 6 not in raw_dots:
                curr_points['Small Straight'] = 15
            if 6 in raw_dots and 1 not in raw_dots:
                curr_points['Big Straight'] = 20

        # Needs to be unable to choose things already played
        # Is this super slow? Might need some numpier solution
        for key, value in self.scoreboard.items():
            if value != self.UNPLAYED_VAL:
                curr_points[key] = self.SCRATCH_VAL

        self.curr_possible_scores = curr_points
        return curr_points

    @abstractmethod
    def play_turn(self):
        ...

    def check_bonus(self) -> int:
        upper_score, upper_is_full = self.get_upper_score()
        if upper_score >= 63:  # This number IS magical
            self.bonus = self.BONUS_VAL
        elif upper_is_full:
            self.bonus = self.SCRATCH_VAL
        return self.bonus

    def get_upper_score(self) -> tuple[int, bool]:
        upper_score = sum([
            val for num in self.NUMS
            if (val := self.scoreboard[num]) != self.SCRATCH_VAL
            and val != self.UNPLAYED_VAL
        ])

        upper_is_full = self.UNPLAYED_VAL not in [
            val for val in [self.scoreboard[num] for num in self.NUMS]
        ]
        return upper_score, upper_is_full

    def get_curr_legal_options(self) -> list[str]:
        self.curr_legal_options = [
            key for key, possible_score in self.curr_possible_scores.items()
            # the possible score will be SCRATCH_VAL if the current scoreboard
            # entry already is taken
            if possible_score != self.SCRATCH_VAL
        ]
        return self.curr_legal_options

    def get_scratch_options(self) -> list[str]:
        self.scratch_options = [
            key for key, score in self.scoreboard.items()
            if score == self.UNPLAYED_VAL
        ]
        return self.scratch_options
