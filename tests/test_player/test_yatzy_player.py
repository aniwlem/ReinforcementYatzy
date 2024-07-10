import pytest

import numpy as np

from reinforcement_yatzy.yatzy.console_player import HumanConsoleYatzyPlayer


@pytest.fixture
def player():
    curr_player = HumanConsoleYatzyPlayer('Player')
    for key in curr_player.scoreboard.keys():
        curr_player.scoreboard[key] = curr_player.UNPLAYED_VAL
    return curr_player


class TestYatzyPlayerFunctions():
    @pytest.mark.parametrize('number', [*range(2, 7)] + [HumanConsoleYatzyPlayer.SCRATCH_VAL])
    def test_no_legal_moves(self, player: HumanConsoleYatzyPlayer, number: int):
        # The dice are all ones, but the only emtpy entry is twos, so the player
        # Will have no legal moves left.
        player.dice = number * np.ones(5, dtype=int)
        for key in player.scoreboard.keys():
            if key != 'Ones':
                player.scoreboard[key] = 1

        player.check_score_current_dice()
        player.get_curr_legal_options()
        print(player.curr_legal_options)
        assert player.curr_legal_options == []

    @pytest.mark.parametrize('number', [*range(2, 7)] + [HumanConsoleYatzyPlayer.SCRATCH_VAL])
    def test_scratch_options(self, player: HumanConsoleYatzyPlayer, number: int):
        # Same as above, but test that the thing that can be scratched is twos
        player.dice = number * np.ones(5, dtype=int)
        for key in player.scoreboard.keys():
            if key != 'Ones':
                player.scoreboard[key] = 1

        player.check_score_current_dice()
        player.get_curr_legal_options()
        player.get_scratch_options()
        print(player.scratch_options)
        assert player.scratch_options == ['Ones']

    @pytest.mark.parametrize('number', range(1, 7))
    def test_get_upper_score(self, player: HumanConsoleYatzyPlayer, number: int):
        for key in player.NUMS:
            player.scoreboard[key] = number

        upper_score, _ = player.get_upper_score()
        assert upper_score == len(player.NUMS) * number

    @pytest.mark.parametrize('should_fill', [True, False])
    def test_is_full(self, player: HumanConsoleYatzyPlayer, should_fill: bool):
        if should_fill:
            for key in player.NUMS:
                player.scoreboard[key] = 10
        _, is_full = player.get_upper_score()
        print(player.scoreboard.values())
        assert is_full == should_fill
        player.scoreboard['One Pair'] = 69

    @pytest.mark.parametrize('index_max', [*range(0, 7)])
    def test_give_bonus(self, player: HumanConsoleYatzyPlayer, index_max: int):
        # As long as the score is above the threshold the bonus should be given,
        # No need to wait until the entire upper section has been filled.
        for i, key in enumerate(player.scoreboard.keys()):
            if i <= index_max:
                player.scoreboard[key] = 100
            else:
                player.scoreboard[key] = player.UNPLAYED_VAL

        player.check_bonus()
        assert player.bonus == player.BONUS_VAL

    def test_no_bonus(self, player: HumanConsoleYatzyPlayer):
        player.scoreboard = {
            key: 2 for key in player.scoreboard.keys()
        }

        player.check_bonus()
        assert player.bonus == player.SCRATCH_VAL

    @pytest.mark.parametrize('index_max', [*range(0, 7)])
    def test_unknown_bonus(self, player: HumanConsoleYatzyPlayer, index_max: int):
        for i, key in enumerate(player.scoreboard.keys()):
            if i < index_max:
                player.scoreboard[key] = 1
            else:
                player.scoreboard[key] = player.UNPLAYED_VAL

        player.check_bonus()
        if index_max == 6:
            assert player.bonus == player.SCRATCH_VAL
        else:
            assert player.bonus == player.UNPLAYED_VAL

    # def test_(self):
    #
    # def test_(self):
    #
    # def test_(self):
    #
    # def test_(self):
    #
    # def test_(self):
    #
    # def test_(self):
