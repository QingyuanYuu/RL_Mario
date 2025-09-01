import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import gym
from numpy import shape
from stable_baselines3 import A2C,PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from test_obs import make_env

#create environment for Mario
def main():
    env = make_env()

    model = PPO("CnnPolicy", env, verbose=1,tensorboard_log='logs',device='mps')
    model.learn(total_timesteps=1)
    print('done learning')

    model.save("ppo_mario")

if __name__ == "__main__":
    main()
