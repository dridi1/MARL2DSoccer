import tensorflow as tf
from typing import Any, NamedTuple
from custom_football_env import FootballEnv as CustomFootballEnv
import dm_env
import numpy as np
from sort_utils import sort_str_num
class EnvironmentSpec(NamedTuple):
    observation_spec: Any
    state_spec: Any
    action_spec: Any
    reward_spec: Any
    action_log_prob_spec: Any

class FootballEnvWrapper:
    """Environment wrapper for 2D Football environment."""

    def __init__(self, render, num_per_team, do_team_switch=False, env_obs_type = "ppo_attention", action_space = "continuous",
                 game_step_lim=400, show_env_feedback=False, include_wait=False, heatmap_save_loc=None,
                 reset_setup="position", dense_shaping_coef: float = 0.0, proximity_coef: float = 0.0,
                 possession_coef: float = 0.0):
        self._num_per_team = num_per_team
       
        self._action_space = action_space
        game_setting = f"{env_obs_type}_state"

        self._environment = CustomFootballEnv(render_game=render, game_setting=game_setting,
                                    players_per_team=[num_per_team, num_per_team],
                                    do_team_switch=do_team_switch, include_wait=include_wait,
                                    game_length=game_step_lim,
                                    game_diff=1.0,
                                    vision_range=np.pi, show_agent_rays=show_env_feedback,
                                    heatmap_save_loc=heatmap_save_loc, reset_setup=reset_setup)
        self.num_channels = CustomFootballEnv.get_num_channels()
        self.prefix = "agent"
        self._dense_shaping_coef = dense_shaping_coef
        self._proximity_coef = proximity_coef
        self._possession_coef = possession_coef
        self._last_shaping_info = None

        # Don't change the game length again.
        self._environment.game_type = "fixed"

    def get_game_length(self):
        return self._environment.game_length

    def reset_game(self):
        timestep, extras = self._environment.reset()
        batch_timestep = self.convert_and_batch_step(timestep, extras)
        self._last_shaping_info = self._environment.get_reward_shaping_info()
        return batch_timestep

    def step(self, actions):
        dict_actions = {}
        for a_i, agent_key in enumerate(self._environment.agent_keys):
            dict_actions[agent_key] = actions[a_i]
        timestep, extras = self._environment.step(dict_actions)
        done = timestep.step_type == dm_env.StepType.LAST
        observations, states, rewards = self.convert_and_batch_step(timestep, extras)

        shaping_info = self._environment.get_reward_shaping_info()

        # Progress shaping: ball x-progress toward opponent goal (agent_0 perspective).
        if self._dense_shaping_coef != 0.0 and self._last_shaping_info is not None:
            try:
                prev_ball_x = float(self._last_shaping_info[f"{self.prefix}_0"][2])
                curr_ball_x = float(shaping_info[f"{self.prefix}_0"][2])
                delta = curr_ball_x - prev_ball_x
                shaped = self._dense_shaping_coef * delta
                for i in range(len(rewards[0])):
                    rewards[0][i] += shaped
                for j in range(len(rewards[1])):
                    rewards[1][j] -= shaped
            except Exception:
                pass

        # Proximity shaping: bonus when team is closer (avg) to ball, penalty if far.
        if self._proximity_coef != 0.0:
            try:
                team_dists = []
                opp_dists = []
                ball_x = float(shaping_info[f"{self.prefix}_0"][2])
                ball_y = float(shaping_info[f"{self.prefix}_0"][3])
                for idx, key in enumerate(sorted(shaping_info.keys(), key=sort_str_num)):
                    p_x, p_y = shaping_info[key][0], shaping_info[key][1]
                    d = np.sqrt((p_x - ball_x) ** 2 + (p_y - ball_y) ** 2)
                    if idx < self._num_per_team:
                        team_dists.append(d)
                    else:
                        opp_dists.append(d)
                if team_dists:
                    team_term = -self._proximity_coef * float(np.mean(team_dists))
                    opp_term = self._proximity_coef * float(np.mean(opp_dists)) if opp_dists else 0.0
                    for i in range(len(rewards[0])):
                        rewards[0][i] += team_term
                    for j in range(len(rewards[1])):
                        rewards[1][j] += opp_term
            except Exception:
                pass

        # Possession shaping: bonus if closest agent is ours, penalty otherwise.
        if self._possession_coef != 0.0:
            try:
                ball_x = float(shaping_info[f"{self.prefix}_0"][2])
                ball_y = float(shaping_info[f"{self.prefix}_0"][3])
                min_dist = None
                min_is_team = False
                for idx, key in enumerate(sorted(shaping_info.keys(), key=sort_str_num)):
                    p_x, p_y = shaping_info[key][0], shaping_info[key][1]
                    d = np.sqrt((p_x - ball_x) ** 2 + (p_y - ball_y) ** 2)
                    if min_dist is None or d < min_dist:
                        min_dist = d
                        min_is_team = idx < self._num_per_team
                if min_dist is not None:
                    if min_is_team:
                        for i in range(len(rewards[0])):
                            rewards[0][i] += self._possession_coef
                        for j in range(len(rewards[1])):
                            rewards[1][j] -= self._possession_coef
                    else:
                        for i in range(len(rewards[0])):
                            rewards[0][i] -= self._possession_coef
                        for j in range(len(rewards[1])):
                            rewards[1][j] += self._possession_coef
            except Exception:
                pass

        self._last_shaping_info = shaping_info

        return observations, states, rewards, done

    def convert_and_batch_step(self, timestep, extras):
        obs_list = [[] for _ in range(len(timestep.observation[f"{self.prefix}_0"].observation))]
        state_list = [[] for _ in range(len(extras["env_states"][f"{self.prefix}_0"]))]
        rewards = []

        for agent in sort_str_num(self._environment.agent_keys):
            obs = timestep.observation[agent].observation
            for i in range(len(obs_list)):
                obs_list[i].append(obs[i])
            state = extras["env_states"][agent]
            if state is not None:
                for i in range(len(state_list)):
                    state_list[i].append(state[i])
            rewards.append(timestep.reward[agent] if timestep.reward is not None else 0.0)
        # Batch the observations
        obs_team = [np.stack(obs_list[i][:self._num_per_team]) for i in range(len(obs_list))]
        obs_opp = [np.stack(obs_list[i][self._num_per_team:]) for i in range(len(obs_list))]
        observations = [obs_team, obs_opp]

        assert len(state_list[0]) == self._num_per_team
        state_team = [np.stack(state_list[i][:self._num_per_team]) for i in range(len(state_list))]
        states = [state_team, None] #state_opp]

        rewards = [rewards[:self._num_per_team], rewards[self._num_per_team:]]
        return observations, states, rewards

    def get_specs(self):
        OBSERVATION_SPEC = self._environment.observation_spec()[f"{self.prefix}_0"].observation
        STATE_SPEC = self._environment.extra_spec()["env_states"][f"{self.prefix}_0"]
        REWARD_SPEC = self._environment.reward_spec()[f"{self.prefix}_0"]
        ACTION_LOG_PROB_SPEC = tf.TensorSpec([], tf.float32)

        if self._action_space=="discrete":
            ACTION_SPEC = tf.TensorSpec([], tf.int32)
            raise NotImplementedError("This has not been used in a long time.")
        else:
            ACTION_SPEC = self._environment.action_spec()[f"{self.prefix}_0"]
        return EnvironmentSpec(observation_spec=[tf.TensorSpec.from_spec(OBSERVATION_SPEC[i]) for i in range(len(OBSERVATION_SPEC))],
                                state_spec=[tf.TensorSpec.from_spec(STATE_SPEC[i]) for i in range(len(STATE_SPEC))],
                                action_spec=tf.TensorSpec.from_spec(ACTION_SPEC),
                                reward_spec=tf.TensorSpec.from_spec(REWARD_SPEC),
                                action_log_prob_spec=tf.TensorSpec.from_spec(ACTION_LOG_PROB_SPEC))
