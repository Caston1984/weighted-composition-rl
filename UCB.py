# import modules
import matplotlib.pyplot as plt
import pandas as pd
import math
import random  # Arm selection based on Exp3 probability
import torch
import numpy as np
from composition_reward import composed_reward


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Ucb:
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
    def __init__(self, k, c, iters):
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
        # Items collected
        self.collected_items = []
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)

    def pull(self):
        # construct UCB values which performs the sqrt part
        ucb_values = np.zeros(self.k)
        for arm in range(self.k):
            ucb_values[arm] = self.k_reward[arm] + self.c * np.sqrt(np.log(self.n) / self.k_n[arm])

        # Select action according to UCB Criteria
        a = np.argmax(ucb_values)

        reward, collected = composed_reward(max_trajectory=50, weight=a/10)

        # Update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n

        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]

        # Update the item collected after the pull
        self.collected_items.append(collected)

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward

    def reset(self):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(self.k)
        self.collected_items = []


k = 11  # number of arms
iters = 1000

#Parameter settings
c = [0, 1, np.sqrt(2), 2, 3]
seeds = [0, 43, 51, 79, 101]

#Create list with 5 variable dataframes for each parameter gamma
data_frames = ['df1', 'df2', 'df3', 'df4', 'df5'] #To store rewards
df_arms = ['df_arms_1', 'df_arms_2', 'df_arms_3', 'df_arms_4', 'df_arms_5']

# Create empty dataframes to store average rewards for each seed per temperature
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df5 = pd.DataFrame()


# Create empty dataframes to store the number of arms collected per temperature
df_arms_1 = pd.DataFrame()
df_arms_2 = pd.DataFrame()
df_arms_3 = pd.DataFrame()
df_arms_4 = pd.DataFrame()
df_arms_5 = pd.DataFrame()


for n_experiment in seeds:
    np.random.seed(n_experiment)
    random.seed(n_experiment)

    for x, df, df_a in zip(c, data_frames, df_arms):
        # Initialize bandits
        ucb = Ucb(k, x, iters)
        ucb_rewards = np.zeros(iters)

        # Run experiments
        ucb.run()

        # Obtain stats
        ucb_rewards = ucb.reward
        ucb_collected = ucb.collected_items
        arms = pd.DataFrame([ucb.k_n - 1])

        #Add results to the DataFrames
        exec('{}[n_experiment] = ucb_rewards'.format(df))
        exec('{} = pd.DataFrame({}).append(arms, ignore_index=True)'.format(df_a, df_a))


#Calculate the average mean and standard deviation curves
mean_1 = df1.mean(axis=1)
std_1 = df1.std(axis=1)
mean_2 = df2.mean(axis=1)
std_2 = df2.std(axis=1)
mean_3 = df3.mean(axis=1)
std_3 = df3.std(axis=1)
mean_4 = df4.mean(axis=1)
std_4 = df4.std(axis=1)
mean_5 = df5.mean(axis=1)
std_5 = df5.std(axis=1)

x = np.arange(len(mean_1))
plt.plot(x, mean_1, 'b-', label='0')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
plt.plot(x, mean_2, 'r-', label='√1')
plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
plt.plot(x, mean_3, 'k-', label='√2')
plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='k', alpha=0.2)
plt.plot(x, mean_4, 'g-', label='√4')
plt.fill_between(x, mean_4 - std_4, mean_4 + std_4, color='g', alpha=0.2)
plt.plot(x, mean_5, 'y-', label='√9')
plt.fill_between(x, mean_5 - std_5, mean_5 + std_5, color='y', alpha=0.2)
plt.ylim(-7.5, -1.0)
plt.legend(title=r'$\c$', loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("11-armed bandits: UCB")
plt.show()

#Calculate the average arms pulled per temperature
mean_arm_1 = df_arms_1.mean()
mean_arm_2 = df_arms_2.mean()
mean_arm_3 = df_arms_3.mean()
mean_arm_4 = df_arms_4.mean()
mean_arm_5 = df_arms_5.mean()


#Make the plot for the mean arms
barWidth = 0.1
# Set position of bar on X axis
br1 = np.arange(len(mean_arm_1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]

#mean_arm_1.index
plt.bar(br1, mean_arm_1, color='b', width=barWidth, edgecolor='grey', label='0')
plt.bar(br2, mean_arm_2, color='r', width=barWidth, edgecolor='grey', label='√1')
plt.bar(br3, mean_arm_3, color='k', width=barWidth, edgecolor='grey', label='√2')
plt.bar(br4, mean_arm_4, color='g', width=barWidth, edgecolor='grey', label='√4')
plt.bar(br5, mean_arm_5, color='y', width=barWidth, edgecolor='grey', label='√9')


# Adding Xticks
plt.title("Weight sampling distribution: UCB")
plt.xlabel('Weight', fontweight='bold', fontsize=15)
plt.ylabel('Average Pulls', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(mean_arm_1))], ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
plt.legend(title=r'$\c$', loc='best')
plt.show()

results = open("UCB_Results", "w")
table = np.column_stack((mean_3, std_3))
np.savetxt(results, table, fmt='%.3e', delimiter="  ")
results.close()
