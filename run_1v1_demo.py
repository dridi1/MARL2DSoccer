"""Run a 1v1 demo using lightweight stubs for missing deps."""
import argparse
import collections
import sys
import types
from typing import Optional

import numpy as np
import tensorflow as tf

# Provide minimal dm_env shim so the environment can construct timesteps.
class _StepType:
    FIRST = 0
    MID = 1
    LAST = 2


_TimeStep = collections.namedtuple("TimeStep", ["step_type", "reward", "discount", "observation"])


def _restart(observation):
    return _TimeStep(step_type=_StepType.FIRST, reward=None, discount=1.0, observation=observation)


dm_env_mod = types.ModuleType("dm_env")
dm_env_mod.StepType = _StepType
dm_env_mod.TimeStep = _TimeStep
dm_env_mod.restart = _restart
sys.modules["dm_env"] = dm_env_mod

# Provide minimal acme specs/types shim for the environment specs.
specs_mod = types.ModuleType("specs")


class _Array:
    def __init__(self, shape, dtype, name=None):
        self.shape = tuple(shape) if isinstance(shape, (list, tuple)) else shape
        self.dtype = dtype
        self.name = name


class _BoundedArray(_Array):
    def __init__(self, shape, dtype, name=None, minimum=None, maximum=None):
        super().__init__(shape, dtype, name)
        self.minimum = np.array(minimum) if minimum is not None else None
        self.maximum = np.array(maximum) if maximum is not None else None


class _DiscreteArray(_Array):
    def __init__(self, num_values, dtype, name=None):
        super().__init__((), dtype, name)
        self.num_values = num_values


specs_mod.Array = _Array
specs_mod.BoundedArray = _BoundedArray
specs_mod.DiscreteArray = _DiscreteArray

atypes_mod = types.ModuleType("types")
atypes_mod.Nest = object
atypes_mod.NestedSpec = object

acme_mod = types.ModuleType("acme")
acme_mod.specs = specs_mod
acme_mod.types = atypes_mod
sys.modules["acme"] = acme_mod
sys.modules["acme.specs"] = specs_mod
sys.modules["acme.types"] = atypes_mod

from env_wrapper import FootballEnvWrapper
from fixed_agent import NaiveAttentionBot
from train_agents import ActorCritic, flatten_team_obs


def main(game_length: int, policy_weights: Optional[str]):
    print(f"Starting 1v1 demo (render window may appear)... game_length={game_length}")
    env = FootballEnvWrapper(num_per_team=1, render=True, include_wait=True, game_step_lim=game_length)

    observations, states, rewards = env.reset_game()
    policy = None
    team_bot = None
    if policy_weights:
        obs_dim = flatten_team_obs(observations[0]).shape[1]
        policy = ActorCritic(obs_dim=obs_dim)
        policy(tf.zeros((1, obs_dim), dtype=tf.float32))  # build weights
        policy.load_weights(policy_weights)
        print(f"Loaded policy weights from {policy_weights}")
    else:
        team_bot = NaiveAttentionBot()

    opp_bot = NaiveAttentionBot()

    done = False
    step = 0
    while not done:
        actions = []
        if policy:
            team_obs_flat = flatten_team_obs(observations[0])
            mu, _, _ = policy(tf.convert_to_tensor(team_obs_flat, dtype=tf.float32))
            team_actions = [np.clip(a.numpy(), -1.0, 1.0) for a in mu]
        else:
            team_actions = team_bot.get_action(observations[0], states[0], add_to_memory=False)

        opp_actions = opp_bot.get_action(observations[1], states[1] if states else None, add_to_memory=False)
        actions.extend(team_actions)
        actions.extend(opp_actions)

        observations, states, rewards, done = env.step(actions)
        step += 1
        if step % 50 == 0:
            print(f"step {step}, rewards {rewards}")
    print("episode finished in", step, "steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a 1v1 demo.")
    parser.add_argument(
        "--game-length",
        type=int,
        default=1200,
        help="Number of environment steps before termination (default: 1200)",
    )
    parser.add_argument(
        "--policy-weights",
        type=str,
        help="Optional path to trained weights to run the learned policy for our team",
    )
    args = parser.parse_args()
    main(args.game_length, args.policy_weights)
