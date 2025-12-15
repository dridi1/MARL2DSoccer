"""Plot training metrics saved in CSV form (episode,reward,length)."""
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_metrics(csv_path: Path) -> Tuple[List[int], List[float], List[int]]:
    episodes: List[int] = []
    rewards: List[float] = []
    lengths: List[int] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not {"episode", "reward", "length"}.issubset(reader.fieldnames or {}):
            raise ValueError("CSV must have columns: episode,reward,length")
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            lengths.append(int(row["length"]))
    return episodes, rewards, lengths


def moving_average(values: List[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=float)
    window = min(window, len(values))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def plot_curves(
    episodes: List[int],
    rewards: List[float],
    lengths: List[int],
    output: Path,
    smooth: int,
    show_lengths: bool,
) -> None:
    plt.figure(figsize=(8, 4.5))

    rewards_to_plot = rewards
    ep_axis = episodes
    label = "Episode reward"

    if smooth and smooth > 1:
        rewards_to_plot = moving_average(rewards, smooth).tolist()
        ep_axis = episodes[smooth - 1 :]
        label = f"Reward (MA{smooth})"

    plt.plot(ep_axis, rewards_to_plot, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training progress")
    plt.grid(True, alpha=0.3)

    if show_lengths:
        # Secondary axis for episode length.
        ax = plt.gca().twinx()
        ax.plot(episodes, lengths, color="tab:orange", alpha=0.6, label="Episode length")
        ax.set_ylabel("Episode length")
        ax.legend(loc="upper right")

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("train_metrics.csv"),
        help="CSV file with columns episode,reward,length (default: train_metrics.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_curve.png"),
        help="Output image file (default: training_curve.png)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Optional moving average window for rewards (0/1 disables smoothing)",
    )
    parser.add_argument(
        "--show-lengths",
        action="store_true",
        help="Overlay episode lengths on a secondary axis",
    )

    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"CSV not found: {args.input}")

    episodes, rewards, lengths = read_metrics(args.input)
    if not episodes:
        raise ValueError("No rows found in CSV")

    plot_curves(episodes, rewards, lengths, args.output, args.smooth, args.show_lengths)


if __name__ == "__main__":
    main()
