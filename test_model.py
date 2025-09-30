import gym
from gym.wrappers import GrayScaleObservation
from numpy import shape
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from test_obs import make_env

if __name__ == '__main__':

    vec_env = SubprocVecEnv([make_env for _ in range(1)])
    vec_env = VecFrameStack(vec_env, 4, channels_order='last')  # 帧叠加

    model = PPO.load('best_model/best_model.zip', env=vec_env)

    obs = vec_env.reset()
    for i in range(10000):
        obs = obs.copy()  # 复制数组以避免负步长问题
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        if done:
          obs = vec_env.reset()
