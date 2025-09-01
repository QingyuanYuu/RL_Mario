import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
import matplotlib.pyplot as plt
from my_wrapper import SkipFrameWrapper


class SkipFrameWrapper(gym.Wrapper):
     def __init__(self, env, skip):
         super().__init__(env)
         self._skip = skip

     def step(self,action):
        obs,reward_total,done,info = None, 0, False, None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            reward_total+=reward
            if done:
                break

        return  obs, reward_total, done, info