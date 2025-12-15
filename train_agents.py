"""Actor-critic training with self-play pool, tqdm logging, and CSV metrics.

Key features:
- Shared Gaussian actor-critic for our team; opponent is a snapshot from a self-play pool (no fixed bot).
- Advantage normalization, time-limit bootstrapping, gradient clipping, and entropy regularization for stability.
- Saves per-episode reward/length for plotting; optional weight export.
"""
import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import trange

from env_wrapper import FootballEnvWrapper

# Training stability knobs
ENTROPY_COEF = 5e-4
GRAD_CLIP_NORM = 0.5


class ActorCritic(tf.keras.Model):
    """Shared-body actor-critic with diagonal Gaussian policy."""

    def __init__(self, obs_dim: int, action_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense2 = tf.keras.layers.Dense(hidden, activation="relu")
        self.mu = tf.keras.layers.Dense(action_dim)
        self.value_head = tf.keras.layers.Dense(1)
        self.log_std = tf.Variable(tf.zeros(action_dim, dtype=tf.float32), trainable=True)

    def call(self, obs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = self.dense1(obs)
        x = self.dense2(x)
        mu = self.mu(x)
        value = tf.squeeze(self.value_head(x), axis=-1)
        log_std = tf.expand_dims(self.log_std, axis=0)  # broadcast over batch
        return mu, log_std, value


def flatten_team_obs(obs_team) -> np.ndarray:
    """Flatten team observations into (num_agents, obs_dim)."""
    if isinstance(obs_team, np.ndarray):
        return obs_team
    parts = [np.asarray(comp) for comp in obs_team]
    flat = np.concatenate([p.reshape(p.shape[0], -1) for p in parts], axis=1)
    return flat


def gaussian_log_prob(actions: tf.Tensor, mu: tf.Tensor, log_std: tf.Tensor) -> tf.Tensor:
    std = tf.exp(log_std)
    log_two_pi = tf.math.log(2.0 * np.pi)
    return -0.5 * tf.reduce_sum(tf.square((actions - mu) / std) + 2 * log_std + log_two_pi, axis=-1)


def clone_weights(model: tf.keras.Model) -> List[np.ndarray]:
    return [w.numpy() for w in model.weights]


def apply_weights(model: tf.keras.Model, weights: List[np.ndarray]) -> None:
    for var, val in zip(model.weights, weights):
        var.assign(val)


class OpponentPool:
    """Stores policy snapshots and samples opponents."""

    def __init__(self, max_size: int, latest_prob: float):
        self.max_size = max_size
        self.latest_prob = latest_prob
        self.snapshots: List[List[np.ndarray]] = []

    def add(self, weights: List[np.ndarray]) -> None:
        if self.max_size <= 0:
            return
        self.snapshots.append([np.copy(w) for w in weights])
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def sample(self) -> Optional[List[np.ndarray]]:
        if not self.snapshots:
            return None
        if np.random.rand() < self.latest_prob:
            return self.snapshots[-1]
        return self.snapshots[np.random.randint(0, len(self.snapshots))]

    def has_snapshots(self) -> bool:
        return len(self.snapshots) > 0


def run_episode(
    env: FootballEnvWrapper,
    policy: ActorCritic,
    opponent_policy: ActorCritic,
    max_steps: int,
) -> Tuple[float, int, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float], bool, float]:
    observations, states, rewards = env.reset_game()
    done = False
    total_reward = 0.0
    steps = 0
    truncated = False

    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    val_buf: List[np.ndarray] = []
    rew_buf: List[float] = []

    while not done and steps < max_steps:
        team_obs_flat = flatten_team_obs(observations[0])
        obs_tensor = tf.convert_to_tensor(team_obs_flat, dtype=tf.float32)
        mu, log_std, values = policy(obs_tensor)
        eps = tf.random.normal(shape=tf.shape(mu))
        actions_tensor = mu + tf.exp(log_std) * eps
        actions_clipped = tf.clip_by_value(actions_tensor, -1.0, 1.0)
        log_probs = gaussian_log_prob(actions_tensor, mu, log_std)

        opp_obs_flat = flatten_team_obs(observations[1])
        opp_mu, _, _ = opponent_policy(tf.convert_to_tensor(opp_obs_flat, dtype=tf.float32))
        opp_actions = [np.clip(a.numpy(), -1.0, 1.0) for a in opp_mu]

        step_actions = [a.numpy() for a in actions_clipped] + opp_actions
        observations, states, rewards, done = env.step(step_actions)

        step_reward = float(np.sum(rewards[0]))  # team reward signal
        total_reward += step_reward
        steps += 1

        obs_buf.append(team_obs_flat)
        act_buf.append(actions_tensor.numpy())
        val_buf.append(values.numpy())
        rew_buf.append(step_reward)

    if not done:
        truncated = True
        # Bootstrap from final state value if time limit hit.
        final_team_obs = flatten_team_obs(observations[0])
        _, _, final_values = policy(tf.convert_to_tensor(final_team_obs, dtype=tf.float32))
        bootstrap_value = float(tf.reduce_mean(final_values))
    else:
        bootstrap_value = 0.0

    return total_reward, steps, obs_buf, act_buf, val_buf, rew_buf, truncated, bootstrap_value


def train(
    episodes: int,
    game_length: int,
    num_per_team: int,
    render: bool,
    log_path: Path,
    smooth: int,
    lr: float,
    gamma: float,
    save_weights: Optional[Path],
    pool_size: int,
    pool_latest_prob: float,
    add_every: int,
    add_threshold: float,
    dense_shaping_coef: float,
    proximity_coef: float,
    possession_coef: float,
):
    env = FootballEnvWrapper(
        num_per_team=num_per_team,
        render=render,
        include_wait=False,
        game_step_lim=game_length,
        dense_shaping_coef=dense_shaping_coef,
        proximity_coef=proximity_coef,
        possession_coef=possession_coef,
    )

    obs0, _, _ = env.reset_game()
    team_size = flatten_team_obs(obs0[0]).shape[0]
    obs_dim = flatten_team_obs(obs0[0]).shape[1]

    policy = ActorCritic(obs_dim=obs_dim)
    opponent_policy = ActorCritic(obs_dim=obs_dim)
    # Build weights
    dummy = tf.zeros((1, obs_dim), dtype=tf.float32)
    policy(dummy)
    opponent_policy(dummy)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    opponent_pool = OpponentPool(max_size=pool_size, latest_prob=pool_latest_prob)
    # Seed pool with initial policy.
    opponent_pool.add(clone_weights(policy))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    recent_rewards: Deque[float] = deque(maxlen=max(1, smooth))
    recent_lengths: Deque[int] = deque(maxlen=max(1, smooth))

    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length"])

        pbar = trange(episodes, desc="Training", dynamic_ncols=True)
        for ep in pbar:
            # Decide opponent: use pool snapshot if available, else mirror current policy.
            snapshot = opponent_pool.sample()
            if snapshot is not None:
                apply_weights(opponent_policy, snapshot)
            else:
                apply_weights(opponent_policy, clone_weights(policy))

            (
                ep_reward,
                ep_len,
                obs_buf,
                act_buf,
                val_buf,
                rew_buf,
                truncated,
                bootstrap_value,
            ) = run_episode(
                env=env,
                policy=policy,
                opponent_policy=opponent_policy,
                max_steps=game_length,
            )

            # Returns and advantages
            returns: List[float] = []
            g = bootstrap_value if truncated else 0.0
            for r in reversed(rew_buf):
                g = r + gamma * g
                returns.append(g)
            returns.reverse()

            obs_batch = np.concatenate(obs_buf, axis=0)
            act_batch = np.concatenate(act_buf, axis=0)
            val_batch = np.concatenate(val_buf, axis=0)

            returns_arr = np.array(returns, dtype=np.float32)
            ret_batch = np.repeat(returns_arr, team_size)
            adv_batch = ret_batch - val_batch[: len(ret_batch)]
            # Advantage normalization
            adv_mean = np.mean(adv_batch)
            adv_std = np.std(adv_batch) + 1e-8
            adv_batch = (adv_batch - adv_mean) / adv_std

            with tf.GradientTape() as tape:
                obs_t = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
                adv_t = tf.convert_to_tensor(adv_batch, dtype=tf.float32)
                ret_t = tf.convert_to_tensor(ret_batch, dtype=tf.float32)

                mu, log_std, values = policy(obs_t)
                log_probs = gaussian_log_prob(tf.convert_to_tensor(act_batch, dtype=tf.float32), mu, log_std)
                values = tf.squeeze(values)

                actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(adv_t))
                critic_loss = tf.reduce_mean(tf.square(ret_t - values))
                entropy = tf.reduce_mean(0.5 * (tf.math.log(2 * np.pi * tf.exp(2 * log_std)) + 1.0))
                loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

            grads = tape.gradient(loss, policy.trainable_variables)
            clipped_grads = [None if g is None else tf.clip_by_norm(g, GRAD_CLIP_NORM) for g in grads]
            optimizer.apply_gradients(zip(clipped_grads, policy.trainable_variables))

            # Pool update (simple gating via recent reward)
            if add_every > 0 and ep > 0 and ep % add_every == 0:
                mean_recent = sum(recent_rewards) / len(recent_rewards) if recent_rewards else ep_reward
                if mean_recent >= add_threshold:
                    opponent_pool.add(clone_weights(policy))

            writer.writerow([ep, ep_reward, ep_len])
            recent_rewards.append(ep_reward)
            recent_lengths.append(ep_len)

            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_len = sum(recent_lengths) / len(recent_lengths)
            pbar.set_postfix(
                reward=f"{ep_reward:.2f}",
                avg_reward=f"{avg_reward:.2f}",
                steps=ep_len,
                avg_len=f"{avg_len:.1f}",
                loss=f"{float(loss):.3f}",
            )

    print(f"Finished {episodes} episodes. Metrics saved to {log_path}")
    if save_weights:
        save_path = save_weights
        if save_path.suffix != ".h5" or not save_path.name.endswith(".weights.h5"):
            save_path = save_path.with_name(save_path.stem + ".weights.h5")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save_weights(save_path)
        print(f"Saved weights to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train agents with self-play pool and log metrics.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to run")
    parser.add_argument("--game-length", type=int, default=1200, help="Max steps per episode")
    parser.add_argument("--num-per-team", type=int, default=1, help="Players per team")
    parser.add_argument("--render", action="store_true", help="Enable rendering (slower)")
    parser.add_argument("--log-path", type=Path, default=Path("train_metrics.csv"), help="Where to save metrics CSV")
    parser.add_argument("--smooth", type=int, default=20, help="Window for progress averages")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--save-weights", type=Path, help="Optional path to save trained weights (auto-append .weights.h5 if missing)")
    parser.add_argument("--pool-size", type=int, default=6, help="Max snapshots to keep in opponent pool")
    parser.add_argument("--pool-latest-prob", type=float, default=0.6, help="Probability to sample latest snapshot vs random older")
    parser.add_argument("--add-every", type=int, default=50, help="Add snapshot to pool every N episodes (if threshold met)")
    parser.add_argument("--add-threshold", type=float, default=-1.0, help="Min recent avg reward to add snapshot to pool")
    parser.add_argument("--dense-shaping-coef", type=float, default=0.02, help="Per-step shaping weight on ball progress (0 to disable)")
    parser.add_argument("--proximity-coef", type=float, default=0.01, help="Per-step shaping weight for average distance to ball (0 to disable)")
    parser.add_argument("--possession-coef", type=float, default=0.005, help="Per-step shaping bonus if closest to ball (0 to disable)")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        game_length=args.game_length,
        num_per_team=args.num_per_team,
        render=args.render,
        log_path=args.log_path,
        smooth=args.smooth,
        lr=args.lr,
        gamma=args.gamma,
        save_weights=args.save_weights,
        pool_size=args.pool_size,
        pool_latest_prob=args.pool_latest_prob,
        add_every=args.add_every,
        add_threshold=args.add_threshold,
        dense_shaping_coef=args.dense_shaping_coef,
        proximity_coef=args.proximity_coef,
        possession_coef=args.possession_coef,
    )


if __name__ == "__main__":
    main()
