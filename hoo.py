"""
This is the preferred one
Implements algorithms detailed in [1]
References:
[1]: Bubeck, Sebastien, et al. "X-armed bandits." Journal of Machine Learning Research 12.May (2011): 1655-1695.
https://github.com/lingchunkai/continuous-armed-bandit/blob/master/hoo.py
"""

import math
import random
import copy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from composition_reward import composed_reward
import gym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Hoo:
    """ Basic HOO descrbed in X-armed bandits [1]
    Assume that search space is in to D-dimensional hypercube [0,1]^D
    Refer to Example 1 of [1] for details about the parameter selection and tree-splitting
    """

    def __init__(self, D, v1, p, fun):
        """ Constructs Hoo empty search tree
        @param D - dimension of hypercube
        @param v1 - tunable parameters --> will affect regret bounds (if it even converges to 0!)
        @param p - tunable parameters --> will affect regret bounds (if it even converges to 0!)
        """
        self.D = D
        self.v1 = v1
        self.p = p
        self.fun = fun
        self.root = Node(np.tile([0.0, 1.0], [D, 1]), None, h=0, nextCut=0)
        self.n = 0

    def Pull(self):
        """ Pull arm
        """
        self.n += 1

        # Find node to pull
        curNode = self.root
        while True:
            # Select best arm
            if curNode.children_B[0] >= curNode.children_B[1]:
                branch = 0
            else:
                branch = 1

            nextNode = curNode.children[branch]

            if nextNode == None: break
            curNode = nextNode

        # Play arm and get reward
        pulled = np.array([random.uniform(curNode.rng[x, 0], curNode.rng[x, 1]) for x in range(D)])
        #reward = self.fun(pulled)
        arm = np.mean(np.array([random.uniform(curNode.rng[x, 0], curNode.rng[x, 1]) for x in range(D)]))
        print(arm)
        reward, collected = composed_reward(max_trajectory=50, weight=arm)

        print('a', curNode.rng)
        print('b', pulled, reward)

        # Create child node
        curNode.children[branch] = curNode.MakeNextNode(branch, self.D)

        # Update all counts and means [T and mu]
        c = curNode.children[branch]
        while not c == None:
            c.T += 1
            c.mu = (1.0 - 1.0 / c.T) * c.mu + reward / c.T
            c = c.parent

        # Update all upper bounds [U]
        stack = [self.root]
        opposite = []
        while len(stack) > 0:
            c = stack.pop(0)
            opposite.append(c)
            c.U = c.mu + math.sqrt((2 * math.log(self.n) / c.T)) + self.v1 * self.p ** c.h
            for z in c.children:
                if z == None: continue
                stack.append(z)

        # Bottom up
        while len(opposite) > 0:
            c = opposite.pop()
            for g in range(2):
                if not c.children[g] == None: c.children_B[g] = c.children[g].B
            c.B = min(c.U, max(c.children_B))

        return pulled, reward, arm


class Node:
    def __init__(self, rng, parent, h=0, nextCut=0):
        self.B = float('inf')
        self.rng = rng
        self.nextCut = nextCut
        self.children_B = [float('inf'), float('inf')]
        self.children = [None, None]
        self.parent = parent
        self.T = 0
        self.mu = 0
        self.U = 0
        self.h = h

    def MakeNextNode(self, pos, D):
        nextRng = copy.deepcopy(self.rng)
        if pos == 0:
            # print self.nextCut
            nextRng[self.nextCut, 0] = (nextRng[self.nextCut, 0] + nextRng[self.nextCut, 1]) / 2.0
        else:

            # print self.nextCut
            nextRng[self.nextCut, 1] = (nextRng[self.nextCut, 0] + nextRng[self.nextCut, 1]) / 2.0
        nextNode = Node(nextRng, self, h=self.h + 1, nextCut=(self.nextCut + 1) % D)
        self.children[pos] = nextNode
        return nextNode


def basic_params():
    """
    Example in Fig 2 of [1]
    """
    a = 2.0  # square error norm
    D = 1  #
    p = 2.0 ** (-a / D)
    b = 1.0
    v1 = b * (2 * math.sqrt(D)) ** a
    objfun = lambda x: float((1.0 / 2.0) * (np.sin(13.0 * x) * np.sin(27.0 * x) + 1))
    noise = lambda: np.random.binomial(1, 0.5)
    fullfun = lambda x: objfun(x) * noise()
    xMaxEReward = 0.867526 / 2.0
    xMax = 0.975599

    return a, D, p, b, v1, fullfun, xMaxEReward, xMax


def advanced_params():
    """
    "Bullseye" reward in [0,1]^2, centered at (.5, .5)
    """

    a = 2.0
    D = 2
    p = 2.0 ** (-a / D)
    b = 1.0
    v1 = b * (2 * math.sqrt(D)) ** a
    objfun = lambda x: math.exp(-0.5 * (np.linalg.norm(x - np.array([0.5, 0.5])) / 0.5) ** 2)
    noise = lambda: np.random.binomial(1, 0.5)
    fullfun = lambda x: objfun(x) * noise()
    xMaxEReward = 0.5
    xMax = [0.5, 0.5]

    return a, D, p, b, v1, fullfun, xMaxEReward, xMax


if __name__ == '__main__':

    # [a,D,p,b,v1,fullfun, xMaxEReward, xMax] = basic_params()
    [a, D, p, b, v1, fullfun, xMaxEReward, xMax] = advanced_params()

    #h = Hoo(D, v1, p, fullfun)
    #arms_pulled = []
    #list_arms = [] #list of the arms used to compose
    #rewards_recieved = []
    #average_regret = []
    #for k in range(1000):
        #pulled, reward, arm = h.Pull()
        #arms_pulled.append(pulled)
        #list_arms.append(arm)
        #regret = xMaxEReward - reward
        #if k == 0:
            #average_regret.append(regret)
        #else:
            #average_regret.append((average_regret[-1] * (k - 1) + regret) / k)

    #print(arms_pulled)
    #plt.plot(average_regret)
    #plt.show()

    # Total mean reward
    mean_reward = 0

    # Parameter settings
    v1 = [0.2, 0.4, 0.5, 0.7, 0.9]
    rho = [0.8, 0.6, 0.5, 0.4, 0.3]
    seeds = [0, 43, 51, 79, 101]

    # Create list with 5 variable dataframes for each temperature
    data_frames = ['df1', 'df2', 'df3', 'df4', 'df5']  # To store rewards
    df_arms = ['df_arms_1', 'df_arms_2', 'df_arms_3', 'df_arms_4', 'df_arms_5']  # To store arms pulled

    # Create empty dataframes to store average rewards for each seed per temperature
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()

    # Create empty dataframes to store the number of arms collected per temperature
    a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, a_16, a_17, a_18, a_19, a_20, a_21, a_22, a_23, a_24, a_25 = [
        [] for _ in range(25)]

    trials = 1000
    a = 0
    for n_experiment in seeds:
        np.random.seed(n_experiment)
        random.seed(n_experiment)

        for v_1, rho_, df, df_a in zip(v1, rho, data_frames, df_arms):
            # Initialisation
            mean_reward = 0
            z_rewards = np.zeros(trials)
            h = Hoo(D, v_1, rho_, fullfun)
            arms_pulled = []
            list_arms = []  # list of the arms used to compose
            rewards_recieved = []
            average_regret = []
            for k in range(trials):
                pulled, reward, arm = h.Pull()
                arms_pulled.append(pulled)
                list_arms.append(arm)
                print(list_arms)

                # Update total
                mean_reward = mean_reward + (reward - mean_reward) / (k + 1)
                z_rewards[k] = mean_reward

            a += 1

            # Add results to the DataFrames
            exec('{}[n_experiment] = z_rewards'.format(df))
            exec('a_{} = list_arms'.format(a))

    print(df1)
    print(a_1)

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
    plt.plot(x, mean_1, 'b-', label=r'$v_1 = 0.2, \rho = 0.8$')
    plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
    plt.plot(x, mean_2, 'r-', label=r'$v_1 = 0.4, \rho = 0.6$')
    plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
    plt.plot(x, mean_3, 'k-', label=r'$v_1 = 0.5, \rho = 0.5$')
    plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='k', alpha=0.2)
    plt.plot(x, mean_4, 'g-', label=r'$v_1 = 0.7, \rho = 0.4$')
    plt.fill_between(x, mean_4 - std_4, mean_4 + std_4, color='g', alpha=0.2)
    plt.plot(x, mean_5, 'y-', label=r'$v_1 = 0.9, \rho = 0.3$')
    plt.fill_between(x, mean_5 - std_5, mean_5 + std_5, color='y', alpha=0.2)
    plt.legend(title=r'$v_1 + \rho$', loc='best')
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("HOO Algorithm")
    plt.show()

    results = open("HOO_Results", "w")
    table = np.column_stack((mean_1, std_1))
    np.savetxt(results, table, fmt='%.3e', delimiter="  ")
    results.close()

    mean_a1 = np.mean(a_1)
    std_a1 = np.std(a_1)
    mean_a6 = np.mean(a_6)
    std_a6 = np.std(a_6)
    mean_a11 = np.mean(a_11)
    std_a11 = np.std(a_11)
    mean_a16 = np.mean(a_16)
    std_a16 = np.std(a_16)
    mean_a21 = np.mean(a_21)
    std_a21 = np.std(a_21)

    x_pos = np.arange(5)  # is the array with the count of the number of bars.
    CTEs = [mean_a1, mean_a6, mean_a11, mean_a16,
            mean_a21]  # is our array which contains the means or heights of the bars.
    error = [std_a1, std_a6, std_a11, std_a16,
             std_a21]  # error sets the heights of the error bars and the standard deviations.

    seed = ['0', '43', '51', '79', '101']

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weight')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seed)
    ax.set_title('HOO Algorithm: ' r'$v_1 = 0.2, \rho = 0.8$'' & Seed = [0, 43, 51, 79, 101]')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('HOO 1.png')
    plt.show()

    mean_a2 = np.mean(a_2)
    std_a2 = np.std(a_2)
    mean_a7 = np.mean(a_7)
    std_a7 = np.std(a_7)
    mean_a12 = np.mean(a_12)
    std_a12 = np.std(a_12)
    mean_a17 = np.mean(a_17)
    std_a17 = np.std(a_17)
    mean_a22 = np.mean(a_22)
    std_a22 = np.std(a_22)
    # print(a_11)

    x_pos = np.arange(5)  # is the array with the count of the number of bars.
    CTEs = [mean_a2, mean_a7, mean_a12, mean_a17,
            mean_a22]  # is our array which contains the means or heights of the bars.
    error = [std_a2, std_a7, std_a12, std_a17,
             std_a22]  # error sets the heights of the error bars and the standard deviations.

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weight')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seed)
    ax.set_title('HOO Algorithm: ' r'$v_1 = 0.4, \rho = 0.6$'' & Seed = [0, 43, 51, 79, 101]')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('HOO 2.png')
    plt.show()

    mean_a3 = np.mean(a_3)
    std_a3 = np.std(a_3)
    mean_a8 = np.mean(a_8)
    std_a8 = np.std(a_8)
    mean_a13 = np.mean(a_13)
    std_a13 = np.std(a_13)
    mean_a18 = np.mean(a_18)
    std_a18 = np.std(a_18)
    mean_a23 = np.mean(a_23)
    std_a23 = np.std(a_23)

    x_pos = np.arange(5)  # is the array with the count of the number of bars.
    CTEs = [mean_a3, mean_a8, mean_a13, mean_a18,
            mean_a23]  # is our array which contains the means or heights of the bars.
    error = [std_a3, std_a8, std_a13, std_a18,
             std_a23]  # error sets the heights of the error bars and the standard deviations.

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weight')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seed)
    ax.set_title('HOO Algorithm: ' r'$v_1 = 0.5, \rho = 0.5$' ' & Seed = [0, 43, 51, 79, 101]')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('HOO 3.png')
    plt.show()

    mean_a4 = np.mean(a_4)
    std_a4 = np.std(a_4)
    mean_a9 = np.mean(a_9)
    std_a9 = np.std(a_9)
    mean_a14 = np.mean(a_14)
    std_a14 = np.std(a_14)
    mean_a19 = np.mean(a_19)
    std_a19 = np.std(a_19)
    mean_a24 = np.mean(a_24)
    std_a24 = np.std(a_24)

    x_pos = np.arange(5)  # is the array with the count of the number of bars.
    CTEs = [mean_a4, mean_a9, mean_a14, mean_a19,
            mean_a24]  # is our array which contains the means or heights of the bars.
    error = [std_a4, std_a9, std_a14, std_a19,
             std_a24]  # error sets the heights of the error bars and the standard deviations.

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weight')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seed)
    ax.set_title('HOO Algorithm: ' r'$v_1 = 0.7, \rho = 0.4$' ' & Seed = [0, 43, 51, 79, 101]')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('HOO 4.png')
    plt.show()

    mean_a5 = np.mean(a_5)
    std_a5 = np.std(a_5)
    mean_a10 = np.mean(a_10)
    std_a10 = np.std(a_10)
    mean_a15 = np.mean(a_15)
    std_a15 = np.std(a_15)
    mean_a20 = np.mean(a_20)
    std_a20 = np.std(a_20)
    mean_a25 = np.mean(a_25)
    std_a25 = np.std(a_25)

    x_pos = np.arange(5)  # is the array with the count of the number of bars.
    CTEs = [mean_a5, mean_a10, mean_a15, mean_a20,
            mean_a25]  # is our array which contains the means or heights of the bars.
    error = [std_a5, std_a10, std_a15, std_a20,
             std_a25]  # error sets the heights of the error bars and the standard deviations.

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weight')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seed)
    ax.set_title('HOO Algorithm: ' r'$v_1 = 0.9, \rho = 0.3$'' & Seed = [0, 43, 51, 79, 101]')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('HOO 5.png')
    plt.show()

    m_1 = np.mean([mean_a1, mean_a6, mean_a11, mean_a16, mean_a21])
    s_1 = np.mean([std_a1, std_a6, std_a11, std_a16, std_a21])
    m_2 = np.mean([mean_a2, mean_a7, mean_a12, mean_a17, mean_a22])
    s_2 = np.mean([std_a2, std_a7, std_a12, std_a17, std_a22])
    m_3 = np.mean([mean_a3, mean_a8, mean_a13, mean_a18, mean_a23])
    s_3 = np.mean([std_a3, std_a8, std_a13, std_a18, std_a23])
    m_4 = np.mean([mean_a4, mean_a9, mean_a14, mean_a19, mean_a24])
    s_4 = np.mean([std_a4, std_a9, std_a14, std_a19, std_a24])
    m_5 = np.mean([mean_a5, mean_a10, mean_a15, mean_a20, mean_a25])
    s_5 = np.mean([std_a5, std_a10, std_a15, std_a20, ])

    x_pos = np.arange(5)  # is the array with the count of the number of bars.
    CTEs = [m_1, m_2, m_3, m_4, m_5]  # is our array which contains the means or heights of the bars.
    error = [s_1, s_2, s_3, s_4, s_5]  # error sets the heights of the error bars and the standard deviations.

    param = [r'$v_1 = 0.2, \rho = 0.8$', r'$v_1 = 0.4, \rho = 0.6$', r'$v_1 = 0.5, \rho = 0.5$', r'$v_1 = 0.7, \rho = 0.4$', r'$v_1 = 0.9, \rho = 0.3$']

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weight')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param)
    ax.set_title('HOO Algorithm: Average arm selected')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('HOO All.png')
    plt.show()

    print('Estimated mean weight', m_1)
    print('Standard Deviation', s_1)




