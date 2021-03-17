import json
import os
from random import random

from gym import Env


class QLearningAgent:
    def __init__(self, env: Env, file_path, epsilon=0.3, alpha=0.5, iterations=1000000):
        self.env = env
        self.q_values = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.actions = list(range(self.env.action_space.n))

        self.file_path = file_path
        self.previously_learnt = False
        if os.path.exists(self.file_path):
            self.previously_learnt = True
            file = open(self.file_path, "r")
            data = json.load(file)
            self.q_values = data
            file.close()

    def q(self, s, a):
        q_value = self.q_values.get(str((tuple(s), a)))
        if q_value is None:
            self.q_values[str((tuple(s), a))] = q_value = 0
        return q_value

    def select_action(self, state):
        if random() <= self.epsilon:
            return self.env.action_space.sample()

        max_q = -10000
        best_current_action = 0
        for action in self.actions:
            q = self.q(state, action)
            if q > max_q:
                max_q = q
                best_current_action = action
        return best_current_action

    def learn(self):
        if self.previously_learnt:
            print("Previously learnt (file {0}). exit learning.".format(self.file_path))

        for iteration in range(self.iterations):
            done = False
            state = self.env.reset()
            count = 0
            while not done:
                action = self.select_action(state)
                new_state, reward, done, info = self.env.step(action)
                q_s_a = self.q(state, action)
                max_q_new_s_a = max([self.q(new_state, a) for a in self.actions])
                self.q_values[str((tuple(state), action))] = round(
                    q_s_a + self.alpha * (reward + max_q_new_s_a - q_s_a), 1
                )
                state = new_state
                count += 1
                if count == 1000:
                    done = True

            if iteration % 1000 == 0:
                print(f"{iteration=}/{self.iterations}, live_for={count}")
                self.alpha *= 2/3
            self.epsilon -= self.epsilon / self.iterations
        self.epsilon = 0
        file = open(self.file_path, "w")
        json.dump(self.q_values, file)
        file.close()
