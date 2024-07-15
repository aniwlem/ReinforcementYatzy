'''
Class made to use to test the abstract base class. Just implements the 
@abstractmethod:s with nonsense so the class can instantiated and used to 
test/use the base class.
'''

import numpy as np

from reinforcement_yatzy.yatzy import ABCYatzyPlayer


class TrainingYatzyPlayer(ABCYatzyPlayer):
    def __init__(self) -> None:
        super().__init__()

    def throw_dice(self, i_dice_throw) -> None:
        new_vals = np.random.randint(1, 7, [len(i_dice_throw)])
        self.dice[i_dice_throw] = new_vals

    def select_dice_to_throw(self) -> list[int]:
        return [2, 3, 5, 7]

    def select_next_entry(self) -> str:
        return 'foo'

    def play_turn(self) -> None:
        return None
