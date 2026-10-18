# import modules
import matplotlib.pyplot as plt
import pandas as pd
"""
Purple circle vs beige square as a function of weights
"""
import gym
import torch
import json
from gym.wrappers.monitor import Monitor

from dqn import ComposedDQN, FloatTensor, get_action
from trainer import load
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame
import numpy as np
from experiment_weighted_or import MaxLength
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ucb_bandit:
    '''
    Upper Confidence Bound Bandit

    Inputs
    ============================================
    k: number of arms (int)
    c:
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0.
        Set to "sequence" for the means to be ordered from
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    def __init__(self, k, c, iters, mu='random'):
        # Number of arms
        self.k = k
        # Exploration parameter
        self.c = c
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)

    def reward_(self, col):
        if (79 <= col[0] <= 82) & (18 <= col[1] <= 21):
            score = 0.9
        elif (0 <= col[0] <= 21) or (83 <= col[1] <= 100):
            score = -1.0
        elif (22 <= col[0] <= 50) or (50 <= col[1] <= 82):
            score = -0.5
        elif (51 <= col[0] <= 78) or (22 <= col[1] <= 49):
            score = -0.2
        else:
            score = 0.2
        return score

    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward + self.c * np.sqrt(
                (np.log(self.n)) / self.k_n))

        collected = np.array(MaxLength.collect(self, 8, 100, 50, a/20))
        print(collected, self.n, a)

        reward = self.reward_(collected)

        # Update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n

        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward

    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(self.k)


k = 21  # number of arms
iters = 1000
ucb_rewards = np.zeros(iters)
# Initialize bandits
ucb = ucb_bandit(k, 0.9, iters)
episodes = 1
# Run experiments

for i in range(episodes):
    ucb.reset('random')
    # Run experiments
    print('EPISODE ---->', i)
    ucb.run()

    # Update long-term averages
    ucb_rewards = ucb_rewards + (
            ucb.reward - ucb_rewards) / (i + 1)

plt.figure(figsize=(12, 8))
plt.plot(ucb_rewards, label="UCB")
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average UCB Rewards after "
          + str(episodes) + " Episodes")
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x=[*range(0, 21, 1)], y=(ucb.k_n - 1), label="Arm Selection")
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Arm")
plt.ylabel("Frequency of arm selection")
plt.title("Arm Selection"
          + str(episodes) + " Episodes")
plt.show()
