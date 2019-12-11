from typing import List

import gym
import numpy as np
from gym import spaces


class SwitchingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, env_index: int):
        super().__init__(env)
        self.env_index = env_index

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return (
            observation,
            reward,
            done,
            {**info, **{"env_index": self.env_index}},
        )

