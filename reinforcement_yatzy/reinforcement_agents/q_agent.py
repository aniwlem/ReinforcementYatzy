import copy
from dataclasses import dataclass
from random import sample
from pathlib import Path

import numpy as np
import torch
from torch import nn

from reinforcement_yatzy.yatzy.base_player import ABCYatzyPlayer


@dataclass
class DiceBufferElement:
    old_dice: torch.Tensor
    new_dice: torch.Tensor
    scoreboard: torch.Tensor
    i_dice_to_throw: torch.Tensor
    throws_left: int
    reward: float


@dataclass
class EntryBufferElement:
    dice: torch.Tensor
    old_scoreboard: torch.Tensor
    new_scoreboard: torch.Tensor
    i_next_entry: int
    reward: float


class DeepQYatzyPlayer(ABCYatzyPlayer):
    '''
    Epsilon-greedy deep-Q agent (Q-learning, not SARSA) yatzy player
    '''

    def __init__(
        self,
        # NOTE: epsilon(epoch) = e**(-gamma * epoch)
        # half life of n games: log(2) / n = 0.69 / n
        gamma_dice: float,
        gamma_entries: float,
        # How much more likely the model is to select entries with a score
        # rather than scratching. Should be close to 0
        invalid_entry_factor: float,
        select_dice_model: nn.Module,
        select_entry_model: nn.Module,
        dice_buffer_size: int,
        entry_buffer_size: int,
        target_change_interval: int,
        batch_size: int,
        penalty_scratch: float,
    ):
        super().__init__()
        self.epsilon_dice = 1
        self.epsilon_entries = 1

        self.gamma_dice = gamma_dice
        self.gamma_entries = gamma_entries

        self.invalid_entry_factor = invalid_entry_factor

        self.dice_buffer_size = dice_buffer_size
        self.entry_buffer_size = entry_buffer_size

        self.dice_buffer: list[DiceBufferElement] = []
        self.entry_buffer: list[EntryBufferElement] = []
        self.target_change_interval = target_change_interval

        self.loss = np.nan

        self.select_dice_model = select_dice_model
        self.target_dice_model = copy.deepcopy(select_dice_model)

        self.select_entry_model = select_entry_model
        self.target_entry_model = copy.deepcopy(select_entry_model)

        # Maybe change these to be tunable from init
        self.dice_optimizer = torch.optim.Adam(
            self.select_dice_model.parameters(),
            lr=1e-3
        )
        self.entry_optimizer = torch.optim.Adam(
            self.select_entry_model.parameters(),
            lr=1e-3
        )

        # want the model to be very sure if it should or shouldn't do something
        self.dice_criterion = nn.CrossEntropyLoss()
        self.entry_criterion = nn.CrossEntropyLoss()

        self.batch_size = batch_size
        self.PENALTY_SCRATCH = penalty_scratch
        self.throws_left = 2
        self.epoch = 0

    def read_network_params(self, *, dice_model_path: Path, entry_model_path: Path) -> None:
        self.select_dice_model.load_state_dict(torch.load(dice_model_path))
        self.target_dice_model.load_state_dict(torch.load(dice_model_path))

        self.select_entry_model.load_state_dict(torch.load(entry_model_path))
        self.target_entry_model.load_state_dict(torch.load(entry_model_path))

        # TODO: stupid API, just have two arguments with paths
    def save_network_params(self, model_paths: dict[str, str]) -> None:
        # TODO: Pickle instead
        torch.save(
            self.select_dice_model.state_dict(), model_paths['dice model']
        )
        torch.save(
            self.select_entry_model.state_dict(), model_paths['entry model']
        )

    def throw_dice(self, i_dice_throw: np.ndarray) -> None:
        new_vals = np.random.randint(1, 7, [np.sum(i_dice_throw, dtype=int)])
        self.dice[i_dice_throw] = new_vals

    def select_dice_to_throw(self) -> np.ndarray:
        if np.random.random() < self.epsilon_dice:
            dice_to_throw = np.random.choice([False, True], [5])
        else:
            curr_dice = torch.tensor(
                self.dice, dtype=torch.float32).unsqueeze(0)

            curr_entries = torch.tensor(
                self.scoreboard, dtype=torch.float32
            ).unsqueeze(0)

            curr_throws_left = torch.tensor(
                self.throws_left, dtype=torch.float32
            ).unsqueeze(0)

            throw_probs = self.select_dice_model(
                curr_dice, curr_entries, curr_throws_left,
            )

            # TODO: threshold should be hyperparam?
            mask_dice_throw = throw_probs.detach().squeeze().numpy() > 0.5
            dice_to_throw = mask_dice_throw

        return dice_to_throw

    def select_next_entry(self) -> int:
        if np.random.random() < self.epsilon_entries:
            is_legal_move = (self.scoreboard == self.UNPLAYED_VAL).astype(int)
            i_legal_moves = np.nonzero(is_legal_move)[0]

            # tweak so non-scratch moves are more likely
            self.check_possible_score_current_dice()
            self.get_curr_legal_options()

            move_is_scratch = self.curr_legal_options[is_legal_move]
            probs = np.ones([self.NUM_ENTRIES])
            probs[move_is_scratch] *= self.invalid_entry_factor
            probs = probs[i_legal_moves]
            probs /= np.sum(probs)

            i_selected_entry = np.random.choice(i_legal_moves, p=probs)
        else:
            curr_dice = torch.tensor(
                self.dice, dtype=torch.float32).unsqueeze(0)

            curr_entries = torch.tensor(
                self.scoreboard, dtype=torch.float32
            ).unsqueeze(0)

            # NOTE: The entry selector could just get the current possible options,
            # instead of the dice
            entry_probs = self.select_entry_model(curr_dice, curr_entries)

            # Use index to save time looking up names in all the following logic
            i_selected_entry = int(torch.argmax(entry_probs))
            is_legal_move = (self.scoreboard == self.UNPLAYED_VAL)
            # If the agent wants to play an illegal move, i.e. select an already
            # occupied entry, take the _legal_ move with the highest prob from the model
            if not is_legal_move[i_selected_entry]:
                # Indices sorted after to how good the agent think it is
                i_sorted_entry_probs = torch.argsort(
                    entry_probs, descending=True
                ).squeeze()

                # Legal moves permuted so best moves are in front, masked so only
                # legal moves are non-zero. To not ignore first entry (index 0)
                # offset all values by one
                offset_sorted_moves_legal_masked = (i_sorted_entry_probs.numpy(
                ) + 1) * is_legal_move[i_sorted_entry_probs.tolist()]

                # Best move is the one the network likes the most, that is legal
                sorted_legal_moves = offset_sorted_moves_legal_masked[
                    np.nonzero(offset_sorted_moves_legal_masked)
                ] - 1  # remove offset

                i_selected_entry = sorted_legal_moves[0]

        return i_selected_entry

    def reinforce(self) -> None:
        # NOTE: wait until both buffers are filled?

        # Don't reinforce until the buffer is filled
        if len(self.dice_buffer) == self.dice_buffer_size:
            batch_indices = sample(
                range(self.dice_buffer_size), self.batch_size
            )
            dice_batch = [self.dice_buffer[i] for i in batch_indices]
            self._reinforce_dice_model(dice_batch)

        if len(self.entry_buffer) == self.entry_buffer_size:
            batch_indices = sample(
                range(self.entry_buffer_size), self.batch_size
            )
            entry_batch = [self.entry_buffer[i] for i in batch_indices]
            self._reinforce_entry_model(entry_batch)

    def _reinforce_dice_model(self, dice_batch: list[DiceBufferElement]) -> None:
        '''
        Does a single q-learning update of the dice_model
        '''
        old_dices = torch.stack([elem.old_dice for elem in dice_batch])
        new_dices = torch.stack([elem.new_dice for elem in dice_batch])
        scoreboards = torch.stack([elem.scoreboard for elem in dice_batch])
        i_dice_to_throws = torch.stack([
            elem.i_dice_to_throw for elem in dice_batch
        ])

        throws_lefts = torch.tensor(
            [elem.throws_left for elem in dice_batch], dtype=torch.float32
        )
        rewards = torch.tensor(
            [elem.reward for elem in dice_batch], dtype=torch.float32
        )

        # Only backprop the dice that were thrown
        backprop_mask = torch.zeros(self.batch_size, self.NUM_DICE)
        for i, i_dice_to_throw in enumerate(i_dice_to_throws):
            backprop_mask[i, i_dice_to_throw.to(torch.int)] = 1

        # To disable the dropout layers and gradients
        self.target_dice_model.eval()
        target_next_reward_preds = self.target_dice_model(
            new_dices,
            scoreboards,
            throws_lefts + 1  # predicts the reward for the next turn
        )
        self.target_dice_model.train()

        future_rewards = torch.max(target_next_reward_preds, dim=1).values
        # There is no future reward for the last throw
        future_rewards[throws_lefts == 0] = 0

        full_rewards = rewards + future_rewards
        full_rewards = full_rewards.unsqueeze(
            1).expand(self.batch_size, self.NUM_DICE)

        self.dice_optimizer.zero_grad()

        self.select_dice_model.eval()
        target_next_reward_preds = self.select_dice_model(
            old_dices, scoreboards, throws_lefts)
        self.select_dice_model.train()

        # Only backprop the dice that a reward was given to
        self.loss = self.dice_criterion(
            backprop_mask * target_next_reward_preds,
            backprop_mask * full_rewards)
        self.loss.backward()
        self.dice_optimizer.step()

    def _reinforce_entry_model(self, entry_batch: list[EntryBufferElement]):
        '''
        Does a single q-learning q-network update of the entry_model
        '''

        dices = torch.stack([elem.dice for elem in entry_batch])
        old_scoreboards = torch.stack([
            elem.old_scoreboard for elem in entry_batch
        ])
        new_scoreboards = torch.stack([
            elem.new_scoreboard for elem in entry_batch
        ])

        i_next_entries = torch.tensor(
            [elem.i_next_entry for elem in entry_batch], dtype=torch.float32
        )
        rewards = torch.tensor(
            [elem.reward for elem in entry_batch], dtype=torch.float32
        )

        backprop_mask = torch.zeros(self.batch_size, self.NUM_ENTRIES)

        for i, i_next_entry in enumerate(i_next_entries):
            backprop_mask[i, i_next_entry.to(torch.int)] = 1

        # To disable the dropout layers and gradients
        self.target_entry_model.eval()
        target_next_reward_preds = self.target_entry_model(
            dices,
            new_scoreboards
        )
        self.target_entry_model.train()

        future_rewards = torch.max(target_next_reward_preds, dim=1).values

        # If the next board is full, don't predict the next reward
        # NOTE: The last scoreboard should get the total score as a reward?
        new_board_is_full = (new_scoreboards == 0).any(dim=1)
        future_rewards[new_board_is_full] = 0

        full_rewards = rewards + future_rewards
        full_rewards = full_rewards.unsqueeze(
            1).expand(self.batch_size, self.NUM_ENTRIES)

        self.entry_optimizer.zero_grad()
        entry_preds = self.select_entry_model(dices, old_scoreboards)
        self.loss = self.entry_criterion(
            backprop_mask * entry_preds,
            backprop_mask * full_rewards)
        self.loss.backward()
        self.entry_optimizer.step()

    def play_turn(self):
        '''
        Plays a single turn, i.e. three dice throws. Stores dice and scoreboard
        states, for each throw into dice_buffer, and for the last throw into the 
        entry_buffer.
        '''
        self.epsilon_dice = np.exp(- self.gamma_dice * self.epoch)
        self.epsilon_entries = np.exp(- self.gamma_entries * self.epoch)

        # The first throw is always of all dice
        i_dice_to_throw = np.ones([self.NUM_DICE], dtype=int)
        self.throw_dice(i_dice_to_throw)
        old_dice = self.dice.copy()
        self.throws_left = 1

        while self.throws_left >= 0:
            i_dice_to_throw = self.select_dice_to_throw()
            self.throw_dice(i_dice_to_throw)
            new_dice = self.dice.copy()

            # the dice buffer is used to train the dice selector, and thus it
            # should try to find the dice to keep for the given dice,
            # scoreboard, and number of throws left.
            self.dice_buffer.append(
                # TODO: embed scoreboard so non-numerics get better
                # representation, and numericals get normalized
                DiceBufferElement(
                    torch.tensor(old_dice, dtype=torch.float32),
                    torch.tensor(new_dice, dtype=torch.float32),
                    torch.tensor(self.scoreboard, dtype=torch.float32),
                    torch.tensor(i_dice_to_throw, dtype=torch.float32),
                    self.throws_left,
                    0,
                )
            )

            if len(self.dice_buffer) > self.dice_buffer_size:
                self.dice_buffer.pop(0)

            old_dice = self.dice.copy()
            self.throws_left -= 1

        old_scoreboard = self.scoreboard
        i_next_entry = self.select_next_entry()
        self.check_possible_score_current_dice()
        self.get_curr_legal_options()

        if i_next_entry in self.curr_legal_options:
            current_point = self.curr_possible_scores[i_next_entry]
            self.scoreboard[i_next_entry] = current_point

        else:
            self.scoreboard[i_next_entry] = self.SCRATCH_VAL
            current_point = self.PENALTY_SCRATCH

        # NOTE: Add the reward to the last dice_buffer element
        # Without this there would be no scores in the dice model
        self.dice_buffer[-1].reward += current_point
        new_scoreboard = self.scoreboard

        self.entry_buffer.append(
            EntryBufferElement(
                torch.tensor(old_dice, dtype=torch.float32),
                torch.tensor(old_scoreboard, dtype=torch.float32),
                torch.tensor(new_scoreboard, dtype=torch.float32),
                i_next_entry,
                current_point,
            )

        )

        if len(self.entry_buffer) > self.entry_buffer_size:
            self.dice_buffer.pop(0)

        self.check_bonus()

        self.reinforce()

    def play_game(self):
        super().play_game()
        self.epoch += 1
