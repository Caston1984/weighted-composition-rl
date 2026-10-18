"""
This implementation doesn't work.
Inspired by
@https://github.com/ntucllab/striatum/blob/408947981e18f22db308d695c8954112ca02041a/striatum/bandit/linthompsamp.py#L126
https://github.com/natetsang/open-rl/blob/d7ec41dc69bd890ccd04b91ee54958b003bb70eb/openrl/algorithms/bandits/cd_linear_ts.py
"""
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import pandas as pd
from composition_reward import composed_reward
import gym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILENAME = "dataset.txt"


def run_lin_ts(k_arms, d, r, steps):
    n_arms = k_arms
    context_dim = d
    ALPHA = 0.001
    DELTA = 0.61
    R = r
    EPSILON = 0.71
    # Initialize vars - some need to be dicts for assigning numpy linalg arrays
    B = np.identity(context_dim)
    mu_hat = np.zeros((context_dim, 1))
    f = np.zeros((context_dim, 1))
    expected_reward = {}
    mean_reward = 0
    avg_reward = np.zeros(steps)

    # Iterate over time T steps
    t = 0
    rewards = []
    cumulative_rewards = 0

    for i in range(steps):
        x1 = np.random.randint(low=0, high=6)
        x2 = np.random.randint(low=0, high=6)
        x3 = np.random.randint(low=0, high=6)
        x4 = np.random.randint(low=0, high=6)
        x5 = np.random.randint(low=0, high=6)
        x6 = np.random.randint(low=0, high=6)
        x7 = np.random.randint(low=0, high=6)
        x8 = np.random.randint(low=0, high=6)
        x9 = np.random.randint(low=0, high=6)
        x10 = np.random.randint(low=0, high=6)
        x11 = np.random.randint(low=0, high=6)
        x12 = np.random.randint(low=0, high=6)
        x13 = np.random.randint(low=0, high=6)
        x14 = np.random.randint(low=0, high=6)
        x15 = np.random.randint(low=0, high=6)
        x16 = np.random.randint(low=0, high=6)
        x17 = np.random.randint(low=0, high=6)
        x18 = np.random.randint(low=0, high=6)
        x19 = np.random.randint(low=0, high=6)
        x20 = np.random.randint(low=0, high=6)
        x21 = np.random.randint(low=0, high=6)
        x22 = np.random.randint(low=0, high=6)
        x23 = np.random.randint(low=0, high=6)
        x24 = np.random.randint(low=0, high=6)
        x25 = np.random.randint(low=0, high=6)
        x26 = np.random.randint(low=0, high=6)
        x27 = np.random.randint(low=0, high=6)
        x28 = np.random.randint(low=0, high=6)
        x29 = np.random.randint(low=0, high=6)
        x30 = np.random.randint(low=0, high=6)
        x31 = np.random.randint(low=0, high=6)
        x32 = np.random.randint(low=0, high=6)
        x33 = np.random.randint(low=0, high=6)
        x34 = np.random.randint(low=0, high=6)
        x35 = np.random.randint(low=0, high=6)
        x36 = np.random.randint(low=0, high=6)
        x37 = np.random.randint(low=0, high=6)
        x38 = np.random.randint(low=0, high=6)
        x39 = np.random.randint(low=0, high=6)
        x40 = np.random.randint(low=0, high=6)
        x41 = np.random.randint(low=0, high=6)
        x42 = np.random.randint(low=0, high=6)
        x43 = np.random.randint(low=0, high=6)
        x44 = np.random.randint(low=0, high=6)
        x45 = np.random.randint(low=0, high=6)
        x46 = np.random.randint(low=0, high=6)
        x47 = np.random.randint(low=0, high=6)
        x48 = np.random.randint(low=0, high=6)
        x49 = np.random.randint(low=0, high=6)
        x50 = np.random.randint(low=0, high=6)

        #Observe the context
        user_context = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35,
                                 x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50])

        #Calculate V
        v = R * np.sqrt(24 / EPSILON * context_dim * np.log(1 / DELTA))
        mu_tilde = np.random.multivariate_normal(mean=mu_hat.flat, cov=v ** 2 * np.linalg.inv(B))[..., np.newaxis]
        #print("MU TILDE", mu_tilde)

        # For each arm, calculate its UCB
        for arm in range(n_arms):
            expected_reward[arm] = np.dot(user_context.T, mu_tilde)

        # Select arm with maximum UCB
        pred_arm = np.argmax(list(expected_reward.values())).item()
        #print("expected_reward", expected_reward)

        #Given the arm, we now use the arm/weight for the composition to obtain the reward
        reward = composed_reward(max_trajectory=50, weight=pred_arm/(n_arms - 1))[0]

        # If the algo_arm matches the data_arm, we can use it

        # Update matrix A and B
        B += np.outer(user_context, user_context)
        f += np.dot(np.reshape(user_context, (-1, 1)), reward)
        mu_hat = np.dot(np.linalg.inv(B), f)

        # Increment the time step
        t += 1

        # Update the mean reward
        mean_reward = mean_reward + (reward - mean_reward) / t
        print(mean_reward)
        avg_reward[i] = mean_reward

        # Update cummulative rewards
        cumulative_rewards += reward
        rewards.append(reward)

        # Calculate CTR for current step t
        #ctr.append(cumulative_rewards / t)
    return avg_reward


if __name__ == "__main__":
    # Parameter settings
    steps = 1000
    R = [0, 0.01, 0.5, 1.0, np.sqrt(2)]
    seeds = [0, 43, 51, 79, 101]

    # Create list with 5 variable dataframes for each temperature
    data_frames = ['df1', 'df2', 'df3', 'df4', 'df5']  # To store rewards

    # Create empty dataframes to store average rewards for each seed per temperature
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()

    for n_experiment in seeds:
        np.random.seed(n_experiment)
        random.seed(n_experiment)

        for r_, df in zip(R, data_frames):
            # Run experiments
            avg_rewards = run_lin_ts(k_arms=101, d=50, r=r_, steps=steps)
            # Add results to the DataFrames
            exec('{}[n_experiment] = avg_rewards'.format(df))

    # Calculate the average mean and standard deviation curves
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
    plt.plot(x, mean_2, 'r-', label='0.01')
    plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
    plt.plot(x, mean_3, 'k-', label='0.5')
    plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='k', alpha=0.2)
    plt.plot(x, mean_4, 'g-', label='1.0')
    plt.fill_between(x, mean_4 - std_4, mean_4 + std_4, color='g', alpha=0.2)
    plt.plot(x, mean_5, 'y-', label='√2')
    plt.fill_between(x, mean_5 - std_5, mean_5 + std_5, color='y', alpha=0.2)
    plt.legend(title=r'$R$', loc='best')
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("LinThompson Sampling Algorithm")
    plt.show()

    results = open("LinTS_Results", "w")
    table = np.column_stack((mean_2, std_2))
    np.savetxt(results, table, fmt='%.3e', delimiter="  ")
    results.close()