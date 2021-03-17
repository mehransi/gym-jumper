from itertools import count

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gym import Env

SEED = 530
torch.manual_seed(SEED)


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, output_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class REINFORCEAgent:
    def __init__(self, env: Env, gamma=0.99, lr=0.01, render=False, max_episodes=None):
        self.env = env
        self.gamma = gamma
        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.render = render
        self.max_episodes = max_episodes

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probabilities = self.policy(state)
        m = Categorical(probabilities)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        eps = np.finfo(np.float32).eps.item()
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def learn(self):
        running_reward = 0
        for episode in count(1):
            state, ep_reward = self.env.reset(), 0
            t = 0
            for i in range(1, 1000):
                t = i
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                if self.render:
                    self.env.render()
                self.policy.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            running_reward = 0.1 * ep_reward + (1 - 0.1) * running_reward
            self.finish_episode()
            if episode % 100 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    episode, ep_reward, running_reward))
            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                return
            if self.max_episodes and episode == self.max_episodes:
                print("reached max_episodes. stop learning!")
                return
