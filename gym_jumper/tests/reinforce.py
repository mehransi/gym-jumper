import time
import gym

from gym_jumper.rl_algorithms import REINFORCEAgent


env = gym.make("Jumper-v0")
agent = REINFORCEAgent(env, gamma=1)

agent.learn()

state = env.reset()

for _ in range(1000):
    action = agent.select_action(state)
    state, reward, done, info = env.step(action)
    if done:
        env.reset()
    env.render()
    time.sleep(0.05)
    print(action, state, reward, done)
env.close()
