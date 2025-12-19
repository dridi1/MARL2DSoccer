"""Evaluate a saved policy against a naive opponent and write episode metrics to CSV."""
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from env_wrapper import FootballEnvWrapper
from fixed_agent import NaiveAttentionBot
from train_agents import ActorCritic, flatten_team_obs


def run_episode(
    env: FootballEnvWrapper,
    policy: ActorCritic,
    opponent: NaiveAttentionBot,
    game_length: int,
) -> Tuple[float, int]:
    """Play one episode headlessly and return (total_team_reward, steps)."""
    observations, states, _ = env.reset_game()
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < game_length:
        team_obs_flat = flatten_team_obs(observations[0])
        mu, _, _ = policy(tf.convert_to_tensor(team_obs_flat, dtype=tf.float32))
        team_actions = [np.clip(a.numpy(), -1.0, 1.0) for a in mu]

        opp_actions = opponent.get_action(observations[1], states[1] if states else None, add_to_memory=False)
        step_actions: List[np.ndarray] = team_actions + opp_actions

        observations, states, rewards, done = env.step(step_actions)
        total_reward += float(np.sum(rewards[0]))
        steps += 1

    return total_reward, steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy and log rewards to CSV.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to saved weights (.weights.h5).")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--game-length", type=int, default=1200, help="Max steps per episode.")
    parser.add_argument("--num-per-team", type=int, default=1, help="Players per team.")
    parser.add_argument("--output", type=Path, default=Path("eval_metrics.csv"), help="CSV output path.")
    parser.add_argument("--render", action="store_true", help="Enable rendering (slower).")
    args = parser.parse_args()

    weights_path = args.weights
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    env = FootballEnvWrapper(
        num_per_team=args.num_per_team,
        render=args.render,
        include_wait=args.render,
        game_step_lim=args.game_length,
    )

    # Build policy to match observation dimensionality.
    obs0, _, _ = env.reset_game()
    obs_dim = flatten_team_obs(obs0[0]).shape[1]
    policy = ActorCritic(obs_dim=obs_dim)
    policy(tf.zeros((1, obs_dim), dtype=tf.float32))  # build layers
    policy.load_weights(weights_path)

    opponent = NaiveAttentionBot()

    results: List[Tuple[int, float, int]] = []
    for ep in range(args.episodes):
        ep_reward, ep_steps = run_episode(env, policy, opponent, args.game_length)
        results.append((ep, ep_reward, ep_steps))
        print(f"Episode {ep}: reward={ep_reward:.3f}, steps={ep_steps}")

    # Write CSV for plotting or reporting.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length"])
        writer.writerows(results)

    rewards = [r for _, r, _ in results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"Wrote {args.output} | avg reward over {len(results)} episodes: {avg_reward:.3f}")
    print("Plot with: python plot_training.py --input eval_metrics.csv --output eval_curve.png --smooth 5")


if __name__ == "__main__":
    main()
