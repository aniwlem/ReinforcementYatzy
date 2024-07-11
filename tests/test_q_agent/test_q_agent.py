from typing import Any
import pytest

import numpy as np
import torch

from reinforcement_yatzy.nn_models.basic_mlp import BaseLineDiceMLP, BaseLineEntryMLP
from reinforcement_yatzy.reinforcement_agents.q_agent import DeepQYatzyPlayer
from reinforcement_yatzy.yatzy.empty_training_player import TrainingYatzyPlayer


class TestQAgent:
    n_dice = 5
    n_entries = TrainingYatzyPlayer.NUM_ENTRIES
    dice_buffer_size = 100
    entry_buffer_size = 100
    target_change_interval = 100
    batch_size = 32
    penalty_scratch = -50

    @pytest.fixture
    def q_agent(self):
        return DeepQYatzyPlayer(
            select_dice_model=BaseLineDiceMLP(
                self.n_dice,
                n_entries=self.n_entries,
                mlp_dims=[2, 3, 5, 7],
            ),
            select_entry_model=BaseLineEntryMLP(
                self.n_dice,
                n_entries=self.n_entries,
                mlp_dims=[2, 3, 5, 7],
            ),
            dice_buffer_size=self.dice_buffer_size,
            entry_buffer_size=self.entry_buffer_size,
            target_change_interval=self.target_change_interval,
            batch_size=self.batch_size,
            penalty_scratch=self.penalty_scratch,
        )

    def get_dice_buffer(
        self,
        batch_size: int
    ) -> list[dict[str, Any]]:
        old_dices = np.random.randint(1, 7, size=[batch_size, self.n_dice])
        new_dices = np.random.randint(1, 7, size=[batch_size, self.n_dice])
        scoreboards = [TrainingYatzyPlayer.scoreboard.copy()
                       for _ in range(batch_size)]
        throws_lefts = np.random.randint(0, 1, size=[batch_size])
        i_dice_to_throws = np.random.choice(
            [0, 1],
            size=[batch_size, self.n_dice],
        )
        rewards = np.random.randint(0, 30, size=[batch_size]) * \
            (1 - throws_lefts)

        dice_buffer_batch = [
            {
                'old_dice': torch.tensor(old_dice, dtype=torch.float32),
                'new_dice': torch.tensor(new_dice, dtype=torch.float32),
                'scoreboard': torch.tensor(list(scoreboard.values()), dtype=torch.float32),
                'i_dice_to_throw': torch.tensor(i_dice_to_throw),
                'throws_left': torch.tensor(throws_left),
                'reward': reward,
            }
            for old_dice, new_dice, scoreboard, i_dice_to_throw, throws_left, reward in
            zip(old_dices, new_dices, scoreboards,
                i_dice_to_throws, throws_lefts, rewards)
        ]

        return dice_buffer_batch

    def get_entry_buffer(
        self,
        batch_size: int
    ) -> list[dict[str, Any]]:
        dices = np.random.randint(1, 7, size=[batch_size, self.n_dice])
        old_scoreboards = [
            TrainingYatzyPlayer.scoreboard.copy() for _ in range(batch_size)
        ]
        new_scoreboards = [
            TrainingYatzyPlayer.scoreboard.copy() for _ in range(batch_size)
        ]

        i_next_entrys = np.random.randint(0, self.n_entries, size=[batch_size])
        rewards = np.random.randint(0, 30, size=[batch_size])

        entry_buffer_batch = [
            {
                'dice': torch.tensor(dice, dtype=torch.float32),
                'old_scoreboard': torch.tensor(list(old_scoreboard.values()), dtype=torch.float32),
                'new_scoreboard': torch.tensor(list(new_scoreboard.values()), dtype=torch.float32),
                'i_next_entry': i_next_entry,
                'reward': reward,
            }
            for dice, old_scoreboard, new_scoreboard, i_next_entry, reward in
            zip(dices, old_scoreboards, new_scoreboards, i_next_entrys, rewards)
        ]
        return entry_buffer_batch

    def test_select_dice(self, q_agent: DeepQYatzyPlayer):
        q_agent.throw_dice(self.n_dice * [1])
        dice_throw_mask = q_agent.select_dice_to_throw()

        assert isinstance(dice_throw_mask, list)

    def test_select_entry(self, q_agent: DeepQYatzyPlayer):
        q_agent.throw_dice(self.n_dice * [1])
        q_agent.check_score_current_dice()
        next_entry = q_agent.select_next_entry()
        assert next_entry in q_agent.scoreboard.keys()

    def test_reinforce_dice(self, q_agent: DeepQYatzyPlayer):
        dice_buffer_batch = self.get_dice_buffer(32)

        weights_before = q_agent.select_dice_model.parameters()
        q_agent._reinforce_dice_model(dice_buffer_batch)
        weights_after = q_agent.select_dice_model.parameters()

        assert weights_before != weights_after

    def test_reinforce_entries(self, q_agent: DeepQYatzyPlayer):
        entry_buffer_batch = self.get_entry_buffer(32)

        weights_before = q_agent.select_entry_model.parameters()
        q_agent._reinforce_entry_model(entry_buffer_batch)
        weights_after = q_agent.select_entry_model.parameters()
        assert weights_before != weights_after

    def test_reinforce(self, q_agent: DeepQYatzyPlayer):
        dice_buffer = self.get_dice_buffer(self.dice_buffer_size)
        entry_buffer = self.get_entry_buffer(self.entry_buffer_size)

        q_agent.dice_buffer = dice_buffer
        q_agent.entry_buffer = entry_buffer

        weights_before = q_agent.select_entry_model.parameters()
        q_agent.reinforce()
        weights_after = q_agent.select_entry_model.parameters()
        assert weights_before != weights_after

    def test_play_turn(self, q_agent: DeepQYatzyPlayer):
        q_agent.play_turn()

    def test_play_game(self, q_agent: DeepQYatzyPlayer):
        q_agent.dice_buffer = []
        q_agent.entry_buffer = []
        for key in q_agent.scoreboard.keys():
            q_agent.scoreboard[key] = q_agent.UNPLAYED_VAL
        q_agent.play_game()
