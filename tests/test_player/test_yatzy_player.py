import pytest

import numpy as np

from reinforcement_yatzy.yatzy import TrainingYatzyPlayer, Entries


@pytest.fixture
def player():
    curr_player = TrainingYatzyPlayer()
    curr_player.reset_scoreboard()
    return curr_player


class TestYatzyPlayerFunctions():
    @pytest.mark.parametrize('number', range(2, 7))
    def test_no_legal_moves(self, player: TrainingYatzyPlayer, number: int):
        # The dice are all twos/threes/something > 1, but the only emtpy entry
        # is ones, so the player will have no legal moves left.
        player.dice = number * np.ones(5, dtype=int)

        player.scoreboard = np.ones_like(player.scoreboard)
        player.scoreboard[Entries.ONES] = player.UNPLAYED_VAL

        player.check_score_current_dice()
        player.get_curr_legal_options()

        assert np.all(player.curr_legal_options == 0)

    @pytest.mark.parametrize('number', [*range(2, 7)] + [TrainingYatzyPlayer.SCRATCH_VAL])
    def test_scratch_options(self, player: TrainingYatzyPlayer, number: int):
        # Same as above, but test that the thing that can be scratched is ones
        player.dice = number * np.ones(5, dtype=int)

        player.scoreboard = np.ones_like(player.scoreboard)
        player.scoreboard[Entries.ONES] = player.UNPLAYED_VAL

        player.check_score_current_dice()
        player.get_curr_legal_options()
        player.get_scratch_options()

        expected_results = np.zeros_like(player.scoreboard)
        expected_results[Entries.ONES] = 1

        print(player.scratch_options)
        print(expected_results)
        assert np.all(player.scratch_options == expected_results)

    @pytest.mark.parametrize('number', range(1, 7))
    def test_get_upper_score(self, player: TrainingYatzyPlayer, number: int):
        for num in player.NUMS:
            player.scoreboard[num] = number

        upper_score, _ = player.get_upper_score()
        assert upper_score == len(player.NUMS) * number

    @pytest.mark.parametrize('should_fill', [True, False])
    def test_is_full(self, player: TrainingYatzyPlayer, should_fill: bool):
        if should_fill:
            for num in player.NUMS:
                player.scoreboard[num] = 10
        _, is_full = player.get_upper_score()
        assert is_full == should_fill

    @pytest.mark.parametrize('index_max', [*range(1, 7)])
    def test_give_bonus(self, player: TrainingYatzyPlayer, index_max: int):
        # As long as the score is above the threshold the bonus should be given,
        # No need to wait until the entire upper section has been filled.
        player.scoreboard[:index_max] = 100
        player.scoreboard[index_max:] = player.UNPLAYED_VAL

        player.check_bonus()
        assert player.bonus == player.BONUS_VAL

    def test_no_bonus(self, player: TrainingYatzyPlayer):
        player.scoreboard = 2 * np.ones_like(player.scoreboard)

        player.check_bonus()
        assert player.bonus == player.SCRATCH_VAL

    @pytest.mark.parametrize('index_max', [*range(0, 7)])
    def test_unknown_bonus(self, player: TrainingYatzyPlayer, index_max: int):
        player.scoreboard[:index_max] = 1
        player.scoreboard[index_max:] = player.UNPLAYED_VAL

        player.check_bonus()
        if index_max == 6:
            assert player.bonus == player.SCRATCH_VAL
        else:
            assert player.bonus == player.UNPLAYED_VAL
