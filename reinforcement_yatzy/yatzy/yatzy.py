from abc import ABC, abstractmethod
import numpy as np


class YatzyPlayer(ABC):
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
    bonus = 0

    def __init__(self) -> None:
        self.dice = np.zeros(self.NUM_DICE, dtype=int)

    def throw_dice(self, ind_throw):
        pass

    @abstractmethod
    def select_dice_to_throw(self) -> list[int]:
        pass

    @abstractmethod
    def select_next_entry(self) -> str:
        pass

    def check_points_of_dice(self) -> None:
        pass

    def play_turn(self):
        pass

    def check_bonus(self):
        pass
