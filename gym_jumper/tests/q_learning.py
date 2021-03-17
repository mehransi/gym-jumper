import gym
import gym_jumper
import time
from gym_jumper.rl_algorithms import QLearningAgent


def run_env():
    env = gym.make('Jumper-v0')
    agent = QLearningAgent(env, file_path="./ql.json", epsilon=0)
    agent.learn()

    state = env.reset()
    for _ in range(1000):
        action = agent.select_action(state)
        print(action)
        state, reward, done, info = env.step(action)  # take a random action
        if done:
            env.reset()
        env.render()
        time.sleep(0.05)
        print(state, reward, done)
    env.close()


if __name__ == "__main__":
    run_env()
