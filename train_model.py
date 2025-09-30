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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback


#create environment for Mario
def linear_schedule(initial_value: float, final_value: float = 0.) :
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func

def main():
    vec_env = SubprocVecEnv([make_env for _ in range(12)])
    vec_env = VecFrameStack(vec_env, 4, channels_order='last')  # 帧叠加
    eval_callback = EvalCallback( best_model_save_path="./best_model/",
                                log_path="./callback_logs/", eval_freq=10000//8,
                                eval_env=vec_env)

    model_params = {
        'learning_rate': linear_schedule(3e-4, 1e-5),  
        'n_steps': 512, 
        'batch_size': 512, 
        'n_epochs': 8, 
        'gamma': 0.95, 
        'gae_lambda': 0.92, 
        'ent_coef': 0.1, 
        'clip_range': linear_schedule(0.25, 0.1),  
        'target_kl': None, 
        'max_grad_norm': 0.5,  
        'vf_coef': 0.75,
        'policy': "CnnPolicy",
        'device': 'cuda',
        'tensorboard_log': './mario_logs/',
        'verbose': 1
    }
    model = PPO(env=vec_env, **model_params)
    # model=PPO.load('./best_model/best_model.zip', env=vec_env,**model_params)
    model.learn(total_timesteps=10e7,callback=[eval_callback])


if __name__ == "__main__":
    main()