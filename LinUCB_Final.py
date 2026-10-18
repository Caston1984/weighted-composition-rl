# https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/

import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import pandas as pd
from composition_reward import composed_reward
import gym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Initiation: Create A and b for that arm
#UCB Calculation: Using x_t context input at time t, we determine the estimate of θhat. Subsequently, we calculate the UCB.
#Reward update: Update information for that arm if it was chosen.

# Create class object for a single linear ucb disjoint arm
class linucb_disjoint_arm():

    def __init__(self, arm_index, d, alpha):
        # Track arm index
        self.arm_index = arm_index

        # Keep track of alpha
        self.alpha = alpha

        # A: (d x d) matrix = D_a.T * D_a + I_d.
        # The inverse of A is used in ridge regression
        self.A = np.identity(d)

        # b: (d x 1) corresponding response vector.
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d, 1])

    def calc_UCB(self, x_array):
        # Find A inverse for ridge regression
        A_inv = np.linalg.inv(self.A)

        # Perform ridge regression to obtain estimate of covariate coefficients theta
        # theta is (d x 1) dimension vector
        self.theta = np.dot(A_inv, self.b)

        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1, 1])

        # Find ucb based on p formulation (mean + std_dev)
        # p is (1 x 1) dimension vector
        p = np.dot(self.theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))

        return p

    def reward_update(self, reward, x_array):
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1, 1])

        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)

        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x


#Here is the Class object for the LinUCB policy for K number of arms. It has two main methods:
#Initiation: Create a list of K linucb_disjoint_arm objects
#Arm selection: Choose arm based on the arm with the highest UCB for a given time step.

class linucb_policy():

    def __init__(self, K_arms, d, alpha):
        self.K_arms = K_arms
        self.linucb_arms = [linucb_disjoint_arm(arm_index=i, d=d, alpha=alpha) for i in range(K_arms)] #there is an error arm_index = i not arm_index = 1, then its only tracking on item with index 1

    def select_arm(self, x_array):
        # Initiate ucb to be 0
        highest_ucb = -1

        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []

        for arm_index in range(self.K_arms):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)

            # If current arm is highest than current highest_ucb
            if arm_ucb > highest_ucb:
                # Set new max ucb
                highest_ucb = arm_ucb

                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            if arm_ucb == highest_ucb:
                candidate_arms.append(arm_index)

        # Choose based on candidate_arms randomly (tie breaker)
        chosen_arm = np.random.choice(candidate_arms)

        return chosen_arm

#The function was designed to return the following:
# - Count of aligned time steps
# - Cumulative rewards
# - Progression log of CTR during aligned time steps
# - LinUCB policy object

def ctr_simulator(K_arms, d, alpha, steps):
    # Initiate policy
    linucb_policy_object = linucb_policy(K_arms=K_arms, d=d, alpha=alpha)

    avg_reward = np.zeros(steps)
    mean_reward = 0
    n=0
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
        data_x_array = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35,
                                 x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50])

        # Find policy's chosen arm based on input covariates at current time step
        arm_index = linucb_policy_object.select_arm(data_x_array)

        #Given the arm, we now use the arm/weight for the composition to obtain the reward
        reward = composed_reward(max_trajectory=50, weight=arm_index/(K_arms - 1))[0]

        #Update counts
        n += 1

        # Update the mean reward
        mean_reward = mean_reward + (reward - mean_reward) / n
        print(mean_reward)
        avg_reward[i] = mean_reward

        cumulative_rewards += reward

    return cumulative_rewards, avg_reward

def simulate(trials):
    #cum_regret = np.zeros((len(algorithms), T + 1))

    # Total mean reward
    mean_reward = 0
    reward = np.zeros(trials)

    # Parameter settings
    c = [0, 1, np.sqrt(2), np.sqrt(3), np.sqrt(4)]
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

        for alpha, df in zip(c, data_frames):
            # Run experiments
            cum_rewards, avg_rewards = ctr_simulator(K_arms=11, d=50, alpha=alpha, steps=trials)
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
    plt.plot(x, mean_2, 'r-', label='√1')
    plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
    plt.plot(x, mean_3, 'k-', label='√2')
    plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='k', alpha=0.2)
    plt.plot(x, mean_4, 'g-', label='√3')
    plt.fill_between(x, mean_4 - std_4, mean_4 + std_4, color='g', alpha=0.2)
    plt.plot(x, mean_5, 'y-', label='√4')
    plt.fill_between(x, mean_5 - std_5, mean_5 + std_5, color='y', alpha=0.2)
    plt.legend(title=r'$c$', loc='best')
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("LinUCB Algorithm")
    plt.show()

    results = open("LinUCB_Results", "w")
    table = np.column_stack((mean_5, std_5))
    np.savetxt(results, table, fmt='%.3e', delimiter="  ")
    results.close()


if __name__ == '__main__':
    simulate(1000)


