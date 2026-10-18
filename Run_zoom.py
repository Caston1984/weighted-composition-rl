"""
Run_zoom.py — Zooming (continuous-armed / Lipschitz bandit) for weighted
composition. Standalone counterpart to the `zooming` entry in
run_experiments.py.

The Zooming implementation is vendored in algorithms.py (from
runninglsy/Lipschitz-bandits-experiment). The weight space is the continuous
interval [0, 1]; the sampled arm is used directly as the composition weight,
and the reward is the shared composed_reward rollout.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame

import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms import Zooming
from composition_reward import composed_reward

MAX_TRAJECTORY = 50


def simulate(trials, out_prefix="Zoom"):
    # Exploration-scale sweep, 5 seeds (matches the paper's protocol).
    parameters = [0, 1, np.sqrt(2), np.sqrt(3), np.sqrt(4)]
    seeds = [0, 43, 51, 79, 101]

    all_curves = {}            # param -> (n_seeds, trials) reward curves
    for c in parameters:
        seed_curves = []
        for s in seeds:
            np.random.seed(s)
            random.seed(s)
            z = Zooming(delta=0.1, T=trials, c=c, nu=1.0)
            z.initialize()
            mean_reward = 0.0
            curve = np.zeros(trials)
            for i in range(trials):
                idx = z.output()
                arm = min(max(z.active_arms[idx], 0.0), 1.0)
                reward, _ = composed_reward(MAX_TRAJECTORY, arm)
                z.observe(i + 1, reward)
                mean_reward += (reward - mean_reward) / (i + 1)
                curve[i] = mean_reward
            seed_curves.append(curve)
        all_curves[c] = np.vstack(seed_curves)

    # Plot mean +/- std per parameter.
    x = np.arange(trials)
    colours = ['b', 'r', 'k', 'g', 'y']
    labels = ['0', '√1', '√2', '√3', '√4']
    plt.figure(figsize=(9, 5))
    for (c, curves), col, lab in zip(all_curves.items(), colours, labels):
        m, sd = curves.mean(axis=0), curves.std(axis=0)
        plt.plot(x, m, col + '-', label=lab)
        plt.fill_between(x, m - sd, m + sd, color=col, alpha=0.2)
    plt.legend(title=r'$\alpha$', loc='best')
    plt.xlabel("Iterations"); plt.ylabel("Average Reward")
    plt.title("Continuous-armed bandit: Zooming")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rewards.png", dpi=120)
    plt.close()

    # Save the best-parameter curve to a results file.
    best_c = max(all_curves, key=lambda k: all_curves[k].mean(axis=0)[-1])
    best = all_curves[best_c]
    with open(f"{out_prefix}_Results", "w") as f:
        np.savetxt(f, np.column_stack((best.mean(axis=0), best.std(axis=0))),
                   fmt='%.3e', delimiter="  ")
    print(f"best alpha={best_c:.4f}  final_avg_reward={best.mean(axis=0)[-1]:.3f}")


if __name__ == '__main__':
    simulate(1000)
