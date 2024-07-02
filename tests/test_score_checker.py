import pytest
# import unittest
import numpy as np
# use console player since the abstract base class can't be instantiate
from reinforcement_yatzy.yatzy.console_player import HumanConsoleYatzyPlayer


@pytest.fixture
def player():
    curr_player = HumanConsoleYatzyPlayer('Player')
    for key in curr_player.scoreboard.keys():
        curr_player.scoreboard[key] = curr_player.UNPLAYED_VAL
    return curr_player


class TestScoreChecker:
    tuple_names: list[str] = [
        'One Pair',
        'Three of a Kind',
        'Four of a Kind',
    ]

    def test_returns_anything(self, player: HumanConsoleYatzyPlayer):
        # make sure it returns anything at all
        player.dice = np.ones(5, dtype=int)
        player.check_points_of_dice()
        assert [
            val for val in player.curr_possible_scores
        ] != []

    # If these work for all numbers, everything else should too, no need to test
    # explicitly later
    @pytest.mark.parametrize('number', range(1, 7))
    def test_numerals(self, player: HumanConsoleYatzyPlayer, number: int):
        player.dice = number * np.ones((5), dtype=int)
        player.check_points_of_dice()
        assert player.curr_possible_scores[
            player.NUMS[number - 1]
        ] == 5 * number

    @pytest.mark.parametrize('number', range(1, 7))
    def test_always_max_pair(self, player: HumanConsoleYatzyPlayer, number: int):
        # The best pair should always be the oned that is selected when there
        # are multiple pairs.
        player.dice = np.array([number, number, 3, 3, 3])
        player.check_points_of_dice()
        assert player.curr_possible_scores['One Pair'] == max(number, 3) * 2

    @pytest.mark.parametrize('number', range(1, 7))
    @pytest.mark.parametrize('tuple_', range(2, 6))
    def test_tuple_scores(self, player: HumanConsoleYatzyPlayer, number: int, tuple_: int):
        player.dice = np.array(
            tuple_ * [number] + (player.NUM_DICE - tuple_) * [1])
        player.check_points_of_dice()

        # check that three of a kind is also a pair etc
        for i in range(2, max(3, tuple_)):
            assert player.curr_possible_scores[
                self.tuple_names[i - 2]
            ] == i * number

    @pytest.mark.parametrize('a', range(1, 7))
    @pytest.mark.parametrize('b', range(1, 7))
    def test_two_pairs(self, player: HumanConsoleYatzyPlayer, a: int, b: int):
        player.dice = np.array([[1] + 2 * [a] + 2 * [b]])
        player.check_points_of_dice()
        if a != b:
            assert player.curr_possible_scores['Two Pairs'] == 2 * a + 2 * b
        else:
            assert player.curr_possible_scores['Two Pairs'] == player.SCRATCH_VAL

    @pytest.mark.parametrize('a', range(1, 7))
    @pytest.mark.parametrize('b', range(1, 7))
    def test_full_house(self, player: HumanConsoleYatzyPlayer, a: int, b: int):
        player.dice = np.array([3 * [a] + 2 * [b]])
        player.check_points_of_dice()
        if a != b:
            assert player.curr_possible_scores['Full House'] == 3 * a + 2 * b
        else:
            assert player.curr_possible_scores['Full House'] == player.SCRATCH_VAL

    @pytest.mark.parametrize('number', range(1, 7))
    def test_straights(self, player: HumanConsoleYatzyPlayer, number: int):
        dice = [*range(1, 7)]
        dice.pop(number - 1)
        player.dice = np.array(dice)
        player.check_points_of_dice()

        print(player.dice)
        print(player.curr_possible_scores['Small Straight'])
        print(player.curr_possible_scores['Big Straight'])
        if number == 1:
            assert player.curr_possible_scores['Small Straight'] == player.SCRATCH_VAL
            assert player.curr_possible_scores['Big Straight'] == 20

        elif number == 6:
            assert player.curr_possible_scores['Small Straight'] == 15
            assert player.curr_possible_scores['Big Straight'] == player.SCRATCH_VAL

        else:
            assert player.curr_possible_scores['Small Straight'] == player.SCRATCH_VAL
            assert player.curr_possible_scores['Big Straight'] == player.SCRATCH_VAL

    @pytest.mark.parametrize('number', range(1, 7))
    @pytest.mark.parametrize('tuple_', range(2, 6))
    def test_yatzy(self, player: HumanConsoleYatzyPlayer, number: int, tuple_: int):
        player.dice = np.array(
            tuple_ * [number] +
            (player.NUM_DICE - tuple_) *
            [(number + 1) % 6 + 1]  # Only yatzy if tuple_ == 5
        )
        player.check_points_of_dice()

        if tuple_ == 5:
            assert player.curr_possible_scores['Yatzy'] == 50
        else:
            assert player.curr_possible_scores['Yatzy'] == player.SCRATCH_VAL
