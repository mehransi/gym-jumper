import os
import gym
import gym_jumper
import time
from stable_baselines3 import PPO


def run_env():
    env = gym.make('Jumper-v0')
    agent = PPO("MlpPolicy", env=env, verbose=1)
    agent.learn(total_timesteps=25000)

    state = env.reset()
    for _ in range(1000):
        action, _states = agent.predict(state, deterministic=True)
        print(action)
        state, reward, done, info = env.step(action)  # take a random action
        if done:
            env.reset()
        env.render()
        time.sleep(0.03)
        print(state, reward, done)
    env.close()


if __name__ == "__main__":
    run_env()
