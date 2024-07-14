import argparse

import numpy as np

from reinforcement_yatzy.nn_models.basic_mlp.baseline_entry_mlp import BaseLineEntryMLP
from reinforcement_yatzy.reinforcement_agents.q_agent import DeepQYatzyPlayer
from reinforcement_yatzy.nn_models.basic_mlp import BaseLineDiceMLP


def main(n_games: int):
    print_interval = 1_000
    dice_model = BaseLineDiceMLP(
        n_dice=DeepQYatzyPlayer.NUM_DICE,
        n_entries=DeepQYatzyPlayer.NUM_ENTRIES,
        mlp_dims=[10, 20, 10],
    )

    entry_model = BaseLineEntryMLP(
        n_dice=DeepQYatzyPlayer.NUM_DICE,
        n_entries=DeepQYatzyPlayer.NUM_ENTRIES,
        mlp_dims=[10, 20, 10],
    )
    agent = DeepQYatzyPlayer(
        select_dice_model=dice_model,
        select_entry_model=entry_model,
        dice_buffer_size=1_000,
        entry_buffer_size=100,
        target_change_interval=1_000,
        batch_size=32,
        penalty_scratch=-30,
    )

    scores = np.zeros([n_games])
    for epoch in range(n_games):
        agent.play_game()
        scores[epoch] = agent.get_total_score()

        if epoch != 0 and epoch % print_interval == 0:
            print(
                f'Epoch {epoch} - Score: np.mean(scores[epoch-print_interval:epoch])')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a q-agent to play yatzy")
    parser.add_argument("n_games", type=int,
                        help="Number of epochs to train")
    args = parser.parse_args()

    main(**vars(args))
