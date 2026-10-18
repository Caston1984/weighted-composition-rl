"""
run_experiments.py — single reproducible driver for the weighted-composition
bandit experiments (companion code to "Weighted composition for entropy-
regularised reinforcement learning", ORiON).

Runs the finite, continuous and contextual bandits against the SHARED reward
(composition_reward.composed_reward), aggregates over seeds, writes tidy CSVs,
and regenerates all figures from the actual runs. One entry point replaces the
scattered per-file __main__ blocks and the manual results-file stitching in the
old Experiment_2/3/4 scripts.

USAGE
-----
    # quick smoke run (tiny, just proves the pipeline end to end)
    python run_experiments.py --smoke

    # full run (paper settings: 5 params x 5 seeds x 1000 iters per algorithm)
    python run_experiments.py --iters 1000 --seeds 0 43 51 79 101

    # a subset of algorithms
    python run_experiments.py --algos softmax ucb hoo --iters 200

Outputs (under ./results_new/):
    rewards_<algo>.csv        mean +/- std average-reward curve per algorithm
    arms_<algo>.csv           weight-selection distribution per algorithm
    summary.csv               best weight +/- std and final avg reward per algo
    fig_contextfree_rewards.png   combined context-free comparison (paper Fig 2)
    fig_contextual_rewards.png    LinUCB vs LinTS (paper Fig 3)
    fig_regret.png                cumulative-regret companion curves
    fig_arms_<algo>.png           corrected arm-selection distributions
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame

import argparse
import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from composition_reward import composed_reward

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_new")
N_WEIGHTS = 11          # discrete weights 0.0, 0.1, ..., 1.0 (finite bandits)
MAX_TRAJECTORY = 50


# ==========================================================================
# Finite-armed bandits (discrete weights): UCB, Softmax, EXP3, Thompson
# ==========================================================================
def run_ucb(c, iters, n_arms=N_WEIGHTS):
    counts = np.zeros(n_arms)
    values = np.zeros(n_arms)
    mean_reward, curve = 0.0, np.zeros(iters)
    for t in range(iters):
        if t < n_arms:
            arm = t  # one initial pull per arm
        else:
            ucb = values + c * np.sqrt(np.log(t + 1) / (counts + 1e-9))
            arm = int(np.argmax(ucb))
        r, _ = composed_reward(MAX_TRAJECTORY, arm / (n_arms - 1))
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]
        mean_reward += (r - mean_reward) / (t + 1)
        curve[t] = mean_reward
    return curve, counts


def run_softmax(tau, iters, n_arms=N_WEIGHTS):
    counts = np.zeros(n_arms)
    values = np.zeros(n_arms)
    mean_reward, curve = 0.0, np.zeros(iters)
    for t in range(iters):
        z = np.sum(np.exp(values / tau))
        probs = np.exp(values / tau) / z
        arm = int(np.random.choice(n_arms, p=probs))
        r, _ = composed_reward(MAX_TRAJECTORY, arm / (n_arms - 1))
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]
        mean_reward += (r - mean_reward) / (t + 1)
        curve[t] = mean_reward
    return curve, counts


def run_exp3(gamma, iters, n_arms=N_WEIGHTS):
    weights = np.ones(n_arms)
    counts = np.zeros(n_arms)
    mean_reward, curve = 0.0, np.zeros(iters)
    for t in range(iters):
        probs = (1 - gamma) * weights / weights.sum() + gamma / n_arms
        arm = int(np.random.choice(n_arms, p=probs))
        r, _ = composed_reward(MAX_TRAJECTORY, arm / (n_arms - 1))
        # EXP3 expects rewards in [0,1]; map the (negative) step reward up.
        scaled = (r + MAX_TRAJECTORY) / MAX_TRAJECTORY
        weights[arm] *= np.exp((gamma / n_arms) * (scaled / probs[arm]))
        counts[arm] += 1
        mean_reward += (r - mean_reward) / (t + 1)
        curve[t] = mean_reward
    return curve, counts


def run_thompson(init_mu, iters, n_arms=N_WEIGHTS):
    mu = np.full(n_arms, init_mu, dtype=float)
    counts = np.zeros(n_arms)
    mean_reward, curve = 0.0, np.zeros(iters)
    for t in range(iters):
        samples = np.random.normal(mu, 1.0)
        arm = int(np.argmax(samples))
        r, _ = composed_reward(MAX_TRAJECTORY, arm / (n_arms - 1))
        counts[arm] += 1
        mu[arm] += (r - mu[arm]) / counts[arm]
        mean_reward += (r - mean_reward) / (t + 1)
        curve[t] = mean_reward
    return curve, counts


# ==========================================================================
# Contextual bandits (random synthetic context, as in the paper Section 5.3):
#   LinUCB and Linear Thompson Sampling
# ==========================================================================
def _random_context(d):
    # paper: synthetic features, scale 0..5 (here d-dim)
    return np.random.randint(0, 6, size=d).astype(float)


def run_linucb(alpha, iters, n_arms=N_WEIGHTS, d=10):
    A = [np.identity(d) for _ in range(n_arms)]
    b = [np.zeros((d, 1)) for _ in range(n_arms)]
    mean_reward, curve = 0.0, np.zeros(iters)
    for t in range(iters):
        x = _random_context(d).reshape(-1, 1)
        p = np.zeros(n_arms)
        for a in range(n_arms):
            A_inv = np.linalg.inv(A[a])
            theta = A_inv @ b[a]
            p[a] = float(theta.T @ x + alpha * np.sqrt(x.T @ A_inv @ x))
        arm = int(np.argmax(p))
        r, _ = composed_reward(MAX_TRAJECTORY, arm / (n_arms - 1))
        A[arm] += x @ x.T
        b[arm] += r * x
        mean_reward += (r - mean_reward) / (t + 1)
        curve[t] = mean_reward
    return curve, None


def run_lints(R, iters, n_arms=N_WEIGHTS, d=10):
    B = np.identity(d)
    f = np.zeros((d, 1))
    mu_hat = np.zeros((d, 1))
    mean_reward, curve = 0.0, np.zeros(iters)
    eps, delta = 0.71, 0.61
    for t in range(iters):
        x = _random_context(d).reshape(-1, 1)
        v = R * np.sqrt(24 / eps * d * np.log(1 / delta)) if R > 0 else 1e-6
        mu_tilde = np.random.multivariate_normal(mu_hat.flatten(),
                                                 v ** 2 * np.linalg.inv(B)).reshape(-1, 1)
        vals = [float(x.T @ mu_tilde) for _ in range(n_arms)]
        arm = int(np.argmax(vals))
        r, _ = composed_reward(MAX_TRAJECTORY, arm / (n_arms - 1))
        B += x @ x.T
        f += r * x
        mu_hat = np.linalg.inv(B) @ f
        mean_reward += (r - mean_reward) / (t + 1)
        curve[t] = mean_reward
    return curve, None


# ==========================================================================
# Experiment registry: name -> (runner, param-list, is_contextual)
# Parameter values follow Table 1 of the paper.
# ==========================================================================
SQRT = np.sqrt
ALGORITHMS = {
    "ucb":     (run_ucb,      [0, SQRT(1), SQRT(2), SQRT(3), SQRT(4)], False),
    "softmax": (run_softmax,  [0.1, 0.2, 0.3, 0.4, 0.5],              False),
    "exp3":    (run_exp3,     [0.1, 0.2, 0.3, 0.4, 0.5],              False),
    "ts":      (run_thompson, [0.1, 0.2, 0.3, 0.4, 0.5],              False),
    "linucb":  (run_linucb,   [0, SQRT(1), SQRT(2), SQRT(3), SQRT(4)], True),
    "lints":   (run_lints,    [0, 0.01, 0.5, 1.0, SQRT(2)],           True),
}


def run_algorithm(name, iters, seeds):
    runner, params, is_ctx = ALGORITHMS[name]
    # best param = the one with the highest final averaged reward
    per_param_curves = {}
    per_param_arms = {}
    for p in params:
        seed_curves = []
        seed_arms = []
        for s in seeds:
            np.random.seed(s)
            random.seed(s)
            curve, arms = runner(p, iters)
            seed_curves.append(curve)
            if arms is not None:
                seed_arms.append(arms)
        per_param_curves[p] = np.vstack(seed_curves)             # (n_seeds, iters)
        if seed_arms:
            per_param_arms[p] = np.vstack(seed_arms).mean(axis=0)  # (n_arms,)
    return per_param_curves, per_param_arms, is_ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=list(ALGORITHMS.keys()))
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 43, 51, 79, 101])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (iters=20, seeds=[0,43]) to prove the pipeline")
    args = ap.parse_args()

    if args.smoke:
        args.iters, args.seeds = 20, [0, 43]

    os.makedirs(OUT, exist_ok=True)
    print(f">> algos={args.algos} iters={args.iters} seeds={args.seeds}")
    print(f">> writing to {OUT}")

    best_curve = {}   # algo -> (mean_curve, std_curve) at its best param
    summary_rows = []

    for name in args.algos:
        print(f"\n== {name} ==")
        curves, arms, is_ctx = run_algorithm(name, args.iters, args.seeds)

        # pick best param by final averaged reward
        best_p, best_final = None, -np.inf
        for p, c in curves.items():
            final = c.mean(axis=0)[-1]
            print(f"   param={p:.4f}  final_avg_reward={final:.3f}")
            if final > best_final:
                best_final, best_p = final, p

        mean_c = curves[best_p].mean(axis=0)
        std_c = curves[best_p].std(axis=0)
        best_curve[name] = (mean_c, std_c, is_ctx)

        # save reward curve CSV
        pd.DataFrame({"iter": np.arange(args.iters),
                      "mean_reward": mean_c,
                      "std_reward": std_c}).to_csv(
            os.path.join(OUT, f"rewards_{name}.csv"), index=False)

        # save + plot arm distribution (finite bandits only), and best weight
        best_weight = np.nan
        if best_p in arms:
            dist = arms[best_p]
            weights = np.linspace(0, 1, len(dist))
            best_weight = float(weights[int(np.argmax(dist))])
            pd.DataFrame({"weight": weights, "avg_pulls": dist}).to_csv(
                os.path.join(OUT, f"arms_{name}.csv"), index=False)
            # corrected arm-distribution figure (fixes the old mean_arm_3 bug)
            plt.figure(figsize=(7, 4))
            plt.bar(weights, dist, width=0.08, color="steelblue", edgecolor="grey")
            plt.xlabel("Weight"); plt.ylabel("Average pulls")
            plt.title(f"Weight-selection distribution: {name.upper()} (best param={best_p:.3f})")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT, f"fig_arms_{name}.png"), dpi=120)
            plt.close()

        summary_rows.append({"algorithm": name, "best_param": best_p,
                             "best_weight": best_weight,
                             "final_avg_reward": best_final, "contextual": is_ctx})

    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "summary.csv"), index=False)

    _plot_reward_comparisons(best_curve, args.iters)
    _plot_regret(best_curve, args.iters)
    print(f"\nDONE. Figures + CSVs in {OUT}")


def _plot_reward_comparisons(best_curve, iters):
    x = np.arange(iters)
    # context-free comparison (paper Fig 2)
    cf = {k: v for k, v in best_curve.items() if not v[2]}
    if cf:
        plt.figure(figsize=(9, 5))
        for name, (m, s, _) in cf.items():
            plt.plot(x, m, label=name.upper())
            plt.fill_between(x, m - s, m + s, alpha=0.15)
        plt.xlabel("Iterations"); plt.ylabel("Average reward")
        plt.title("Context-free bandits: average reward (best param per algorithm)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig_contextfree_rewards.png"), dpi=120)
        plt.close()
    # contextual comparison (paper Fig 3)
    ctx = {k: v for k, v in best_curve.items() if v[2]}
    if ctx:
        plt.figure(figsize=(9, 5))
        for name, (m, s, _) in ctx.items():
            plt.plot(x, m, label=name.upper())
            plt.fill_between(x, m - s, m + s, alpha=0.15)
        plt.xlabel("Iterations"); plt.ylabel("Average reward")
        plt.title("Contextual bandits: average reward (best param per algorithm)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig_contextual_rewards.png"), dpi=120)
        plt.close()


def _plot_regret(best_curve, iters):
    # Simple companion regret plot: regret vs the best achieved mean reward
    # across all algorithms (a proxy for the unknown optimal). Cumulative.
    x = np.arange(iters)
    best_level = max(m[-1] for m, _, _ in best_curve.values())
    plt.figure(figsize=(9, 5))
    for name, (m, _s, _ctx) in best_curve.items():
        inst_regret = np.maximum(best_level - m, 0.0)
        plt.plot(x, np.cumsum(inst_regret), label=name.upper())
    plt.xlabel("Iterations"); plt.ylabel("Cumulative regret (proxy)")
    plt.title("Cumulative regret vs best achieved mean reward")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_regret.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
