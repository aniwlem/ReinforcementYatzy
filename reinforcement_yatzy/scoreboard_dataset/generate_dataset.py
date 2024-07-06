import argparse
from pathlib import Path

from reinforcement_yatzy.scoreboard_dataset.scoreboard_generator import ScoreboardGenerator


def main(save_path: Path, batch_size: int, n_chunks: int):
    generator = ScoreboardGenerator(save_path)
    generator.append_chunks(batch_size, n_chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generates a dataset of yatzy scoreboards")
    parser.add_argument("save_path", type=Path,
                        help="Path to save the dataset.")
    parser.add_argument("batch_size", type=int,
                        help="Number of scoreboards to generate and keep in memory at any given time")
    parser.add_argument("n_chunks", type=int,
                        help="Number of times batch_size scoreboards should be created and appended to the dataset")
    args = parser.parse_args()

    main(**vars(args))
