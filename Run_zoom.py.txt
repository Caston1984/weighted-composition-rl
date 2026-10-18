#https://github.com/runninglsy/Lipschitz-bandits-experiment/blob/master/algorithms.py
import gym
import random
import numpy as np
from scipy.stats import pareto
from algorithms import Zooming
import torch
from experiment_weighted_or import MaxLength
import matplotlib.pyplot as plt
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def simulate(trials):
    #cum_regret = np.zeros((len(algorithms), T + 1))

    # Total mean reward
    mean_reward = 0
    reward = np.zeros(trials)

    # Parameter settings
    parameters = [0, 1, np.sqrt(2), np.sqrt(3), np.sqrt(4)]
    seeds = [0, 43, 51, 79, 101]

    # Create list with 5 variable dataframes for each temperature
    data_frames = ['df1', 'df2', 'df3', 'df4', 'df5']  # To store rewards
    df_arms = ['df_arms_1', 'df_arms_2', 'df_arms_3', 'df_arms_4', 'df_arms_5'] # To store arms pulled

    # Create empty dataframes to store average rewards for each seed per temperature
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()

    # Create empty dataframes to store the number of arms collected per temperature
    a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, a_16, a_17, a_18, a_19, a_20, a_21, a_22, a_23, a_24, a_25 = [[] for _ in range(25)]

    a=0
    for n_experiment in seeds:
        np.random.seed(n_experiment)
        random.seed(n_experiment)

        for c, df, df_a in zip(parameters, data_frames, df_arms):
            # Initialize bandits
            z = Zooming(c)
            z_rewards = np.zeros(trials)

            for i in range(trials):
                # Run experiments
                idx = z.output()
                arm = z.active_arms[idx]
                #tried_arms.append(arm)
                #print('arm', arm)
                #print('parameter',c,'active arms', z.active_arms)

                # Get the reward and item collected
                m = MaxLength(gym.Wrapper, 50)
                reward, collected = m.reward(max_trajectory=50, weight=arm)
                #print(c, i, reward)

                # Update total
                mean_reward = mean_reward + (reward - mean_reward)/(i+1)
                z_rewards[i] = mean_reward

                # inst_regret[i, t] = min(abs(arm - 0.4), abs(arm - 0.8))
                # y = a - min(abs(arm - 0.4), abs(arm - 0.8)) + pareto.rvs(alpha) - alpha / (alpha - 1)
                z.observe(i, reward)
            a += 1


            # Add results to the DataFrames
            exec('{}[n_experiment] = z_rewards'.format(df))
            exec('a_{} = a_{}.append(z.active_arms)'.format(a, a))
            #exec('{} = pd.DataFrame({}).append(tried_arms), ignore_index=True)'.format(df_a, df_a))





    #for trial in range(trials):
    #    inst_regret = np.zeros((len(algorithms), T + 1))
    #    for alg in algorithms:
    #        alg.initialize()

    #    for t in range(1, T + 1):
    #        for i, alg in enumerate(algorithms):
    #            idx = alg.output()
    #            arm = alg.active_arms[idx]
    #            # Get the reward and item collected
    #            c = MaxLength(gym.Wrapper, 50)
    #            reward, collected = c.reward(max_trajectory=50, weight=arm)
    #            print(reward)
                #inst_regret[i, t] = min(abs(arm - 0.4), abs(arm - 0.8))
                #y = a - min(abs(arm - 0.4), abs(arm - 0.8)) + pareto.rvs(alpha) - alpha / (alpha - 1)
    #            alg.observe(t, reward)

      #  cum_regret += np.cumsum(inst_regret, axis=-1)
    #return cum_regret / trials

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
    plt.legend(title=r'$\alpha$', loc='best')
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("11-armed bandits: Zooming Algorithm")
    plt.show()


    results = open("Zoom_Results", "w")
    table = np.column_stack((mean_4, std_4))
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

    x_pos = np.arange(5) #is the array with the count of the number of bars.
    CTEs = [mean_a1, mean_a6, mean_a11, mean_a16, mean_a21] # is our array which contains the means or heights of the bars.
    error = [std_a1, std_a6, std_a11, std_a16, std_a21] # error sets the heights of the error bars and the standard deviations.

    seed = ['0', '43', '51', '79', '101']

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weight')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seed)
    ax.set_title('Zooming Algorithm: ' r'$\alpha$''= 0 & Seed = [0, 43, 51, 79, 101]')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('alpha 0.png')
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
    #print(a_11)

    x_pos = np.arange(5) #is the array with the count of the number of bars.
    CTEs = [mean_a2, mean_a7, mean_a12, mean_a17, mean_a22] # is our array which contains the means or heights of the bars.
    error = [std_a2, std_a7, std_a12, std_a17, std_a22] # error sets the heights of the error bars and the standard deviations.

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weight')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seed)
    ax.set_title('Zooming Algorithm: ' r'$\alpha$''= √1 & Seed = [0, 43, 51, 79, 101]')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('alpha √1.png')
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
    ax.set_title('Zooming Algorithm: ' r'$\alpha$''= √2 & Seed = [0, 43, 51, 79, 101]')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('alpha √2.png')
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
    ax.set_title('Zooming Algorithm: ' r'$\alpha$''= √3 & Seed = [0, 43, 51, 79, 101]')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('alpha √3.png')
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
    ax.set_title('Zooming Algorithm: ' r'$\alpha$''= √4 & Seed = [0, 43, 51, 79, 101]')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('alpha √4.png')
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

    alpha = ['0', '√1', '√2', '√3', '√4']

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weight')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(alpha)
    ax.set_title('Zooming Algorithm: Average arm selected')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('alpha.png')
    plt.show()


def run_experiment(trials):
    # configure parameters of experiments
    #T = 20
    #trials = 40
    #delta = 0.1
    #alpha = 3.1
    #epsilon = 1
    simulate(trials)

    # compute upper bounds for moments of different orders
    #a_hat = max(abs(a), abs(a - 0.4))
    #sigma_second = max(alpha / ((alpha - 1) ** 2 * (alpha - 2)), 1 / (36 * np.sqrt(2)))
    #nu_second = max(a_hat ** 2 + sigma_second, np.power(12 * np.sqrt(2), -(1 + epsilon)))
    #nu_third = a_hat ** 3 + 2 * alpha * (alpha + 1) / (
            #(alpha - 1) ** 3 * (alpha - 2) * (alpha - 3)) + 3 * a_hat * sigma_second

    # simulate
    c_zooming = 0.01  # searched within {1, 0.1, 0.01} and `0.01` is the best choice
    c_ADTM = 0.1  # searched within {1, 0.1, 0.01} and `0.1` is the best choice
    c_ADMM = 0.1  # searched within {1, 0.1, 0.01} and `0.1` is the best choice
    #algorithms = [Zooming()] #, nu_third
    #algorithms = [Zooming(delta, T, c_zooming, nu_third), ADTM(delta, T, c_ADTM, nu_second, epsilon),
                  #ADMM(delta, T, c_ADMM, sigma_second, epsilon)]
    #cum_regret = simulate(algorithms, a, alpha, T, trials)

    # plot figure
    #plt.figure(figsize=(7, 4))
    #plt.locator_params(axis='x', nbins=5)
    #plt.locator_params(axis='y', nbins=5)
    #names = [f'{alg.__class__.__name__}' for alg in algorithms]
    #linestyles = ['-', '--', '-.']
    #for result, name, linestyle in zip(cum_regret, names, linestyles):
    #    plt.plot(result, label=name, linewidth=2.0, linestyle=linestyle)
    #plt.legend(loc='upper left', frameon=True, fontsize=10)
    #plt.xlabel('t', labelpad=1, fontsize=15)
    #plt.ylabel('cumulative regret', fontsize=15)
    #plt.xticks(fontsize=15)
    #plt.yticks(fontsize=15)
    #plt.savefig(f'cum_regret_{a}.png', dpi=500, bbox_inches='tight')
    #plt.show()


if __name__ == '__main__':
    run_experiment(1000)
    #run_experiment(a=-2)

