import gym
import gym_jumper
import time


def run_env():
    env = gym.make('Jumper-v0')
    state = env.reset()
    print(state)
    for _ in range(1000):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)  # take a random action
        if done:
            env.reset()
        env.render()
        time.sleep(0.03)
        print(action, state, reward, done)
    env.close()


if __name__ == "__main__":
    run_env()
