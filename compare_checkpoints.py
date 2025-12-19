"""Evaluate multiple policy checkpoints and plot mean reward vs checkpoint.

This script loads each `*.weights.h5` file, evaluates it for N episodes against a
naive scripted opponent, writes a CSV summary, and saves a comparison plot.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from env_wrapper import FootballEnvWrapper
from fixed_agent import NaiveAttentionBot
from train_agents import ActorCritic, flatten_team_obs


def _checkpoint_id(path: Path) -> Optional[int]:
    """Extract a numeric checkpoint id from filenames like ppo_actor20000.weights.h5."""
    match = re.search(r"(\d+)(?=\.weights\.h5$)", path.name)
    if match:
        return int(match.group(1))
    # treat non-numbered as 0 (e.g., ppo_actor.weights.h5)
    if path.name.endswith(".weights.h5"):
        return 0
    return None


def run_episode(
    env: FootballEnvWrapper, policy: ActorCritic, opponent: NaiveAttentionBot, game_length: int
) -> Tuple[float, int, Dict[str, int]]:
    observations, states, _ = env.reset_game()
    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < game_length:
        team_obs_flat = flatten_team_obs(observations[0])
        mu, _, _ = policy(tf.convert_to_tensor(team_obs_flat, dtype=tf.float32))
        team_actions = [np.clip(a.numpy(), -1.0, 1.0) for a in mu]

        opp_actions = opponent.get_action(observations[1], states[1] if states else None, add_to_memory=False)
        observations, states, rewards, done = env.step(team_actions + opp_actions)
        total_reward += float(np.sum(rewards[0]))
        steps += 1
    goals_dict = {"blue": 0, "red": 0}
    inner_env = getattr(env, "_environment", None)
    if inner_env is not None and hasattr(inner_env, "goals"):
        goals_dict = dict(inner_env.goals)
    return total_reward, steps, goals_dict


def eval_weights(
    env: FootballEnvWrapper,
    policy: ActorCritic,
    weights_path: Path,
    episodes: int,
    game_length: int,
    seed: Optional[int],
) -> Tuple[float, float, float, float, float, float, float, float]:
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    policy.load_weights(weights_path)
    opponent = NaiveAttentionBot()
    rewards: List[float] = []
    wins = 0
    draws = 0
    losses = 0
    goals_for = 0
    goals_against = 0
    for _ in range(episodes):
        r, _, goals = run_episode(env, policy, opponent, game_length)
        rewards.append(r)
        # Our controlled agents are agent_0.. agent_(num_per_team-1), which are always the blue team.
        gf = int(goals.get("blue", 0))
        ga = int(goals.get("red", 0))
        goals_for += gf
        goals_against += ga
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1
    mean = float(np.mean(rewards)) if rewards else 0.0
    std = float(np.std(rewards)) if rewards else 0.0
    denom = max(1, episodes)
    win_rate = wins / denom
    draw_rate = draws / denom
    loss_rate = losses / denom
    avg_goal_diff = (goals_for - goals_against) / denom
    avg_gf = goals_for / denom
    avg_ga = goals_against / denom
    return mean, std, win_rate, draw_rate, loss_rate, avg_goal_diff, avg_gf, avg_ga


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple checkpoints by mean eval reward.")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory containing *.weights.h5 checkpoints (default: weights/).",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes per checkpoint.")
    parser.add_argument("--game-length", type=int, default=1200, help="Max steps per episode.")
    parser.add_argument("--num-per-team", type=int, default=1, help="Players per team.")
    parser.add_argument("--seed", type=int, help="Optional RNG seed for repeatability.")
    parser.add_argument("--output-csv", type=Path, default=Path("checkpoint_compare.csv"), help="Summary CSV output.")
    parser.add_argument("--output-plot", type=Path, default=Path("checkpoint_compare.png"), help="Plot output (PNG).")
    parser.add_argument(
        "--output-bar-plot",
        type=Path,
        default=Path("checkpoint_compare_bar.png"),
        help="Poster-friendly bar chart output (PNG).",
    )
    args = parser.parse_args()

    weights_dir: Path = args.weights_dir
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights dir not found: {weights_dir}")

    checkpoints = sorted(weights_dir.glob("*.weights.h5"), key=lambda p: (_checkpoint_id(p) is None, _checkpoint_id(p) or 0, p.name))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {weights_dir} (expected *.weights.h5)")

    env = FootballEnvWrapper(
        num_per_team=args.num_per_team,
        render=False,
        include_wait=False,
        game_step_lim=args.game_length,
    )

    # Build policy to match the environment observation dimensionality.
    obs0, _, _ = env.reset_game()
    obs_dim = flatten_team_obs(obs0[0]).shape[1]
    policy = ActorCritic(obs_dim=obs_dim)
    policy(tf.zeros((1, obs_dim), dtype=tf.float32))

    rows = []
    for ckpt in checkpoints:
        ckpt_id = _checkpoint_id(ckpt)
        mean, std, win_rate, draw_rate, loss_rate, avg_goal_diff, avg_gf, avg_ga = eval_weights(
            env, policy, ckpt, args.episodes, args.game_length, args.seed
        )
        rows.append(
            (
                ckpt.name,
                ckpt_id if ckpt_id is not None else "",
                mean,
                std,
                win_rate,
                draw_rate,
                loss_rate,
                avg_goal_diff,
                avg_gf,
                avg_ga,
            )
        )
        print(
            f"{ckpt.name}: mean={mean:.3f}, std={std:.3f}, win%={win_rate*100:.1f}, goal_diff={avg_goal_diff:.2f} (episodes={args.episodes})"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "checkpoint",
                "step",
                "mean_reward",
                "std_reward",
                "win_rate",
                "draw_rate",
                "loss_rate",
                "avg_goal_diff",
                "avg_goals_for",
                "avg_goals_against",
            ]
        )
        w.writerows(rows)

    # Plot: mean reward vs checkpoint step
    steps = []
    means = []
    stds = []
    labels = []
    win_rates = []
    for name, step, mean, std, win_rate, *_rest in rows:
        labels.append(name)
        steps.append(int(step) if str(step).isdigit() else 0)
        means.append(float(mean))
        stds.append(float(std))
        win_rates.append(float(win_rate))

    plt.figure(figsize=(8, 4.5))
    if len(set(steps)) > 1:
        plt.errorbar(steps, means, yerr=stds, fmt="-o", capsize=4)
        plt.xlabel("Checkpoint step")
    else:
        x = np.arange(len(labels))
        plt.errorbar(x, means, yerr=stds, fmt="-o", capsize=4)
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.xlabel("Checkpoint")
    plt.ylabel("Mean eval reward")
    plt.title(f"Checkpoint comparison (N={args.episodes} episodes)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_plot)

    # Poster-friendly bar chart.
    x = np.arange(len(labels))
    plt.figure(figsize=(9, 4.8))
    plt.bar(x, means, yerr=stds, capsize=5, color="tab:blue", alpha=0.85)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Mean eval reward")
    plt.title(f"Checkpoint comparison (N={args.episodes} episodes)")
    plt.grid(True, axis="y", alpha=0.3)
    for i, (m, w) in enumerate(zip(means, win_rates)):
        offset = 0.2 if m >= 0 else -0.2
        plt.text(i, m + offset, f"win {w*100:.0f}%", ha="center", va="bottom" if m >= 0 else "top", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output_bar_plot)
    print(f"Wrote {args.output_csv}")
    print(f"Saved plot to {args.output_plot}")
    print(f"Saved bar plot to {args.output_bar_plot}")


if __name__ == "__main__":
    main()
