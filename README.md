# Weighted Composition for Entropy-Regularised RL — Bandit Experiments

Companion code for the paper **"Weighted composition for entropy-regularised
reinforcement learning"** (ORiON, [DOI 10.5784/41-2-784](https://doi.org/10.5784/41-2-784)).

The project studies how an agent can **compose previously-learned skills** by
choosing the *weights* of the composition, and frames that weight-selection as a
**multi-armed bandit** problem. Pre-trained entropy-regularised skills (collect
*purple circle*, collect *beige square*) are blended with weights `[w, 1-w]`; a
bandit algorithm searches for the weight that maximises episodic return in a
high-dimensional pixel gridworld.

Three bandit settings are compared, following the paper:

| Setting | Algorithms | Weight space |
|---|---|---|
| Finite (discrete) | UCB, Softmax, EXP3, Thompson Sampling | 11 discrete weights 0.0–1.0 |
| Continuous | HOO, Zooming | continuous `w ∈ [0,1]` |
| Contextual | LinUCB, Linear Thompson Sampling | discrete weights + random context |

## Quick start

```bash
pip install -r requirements.txt

# prove the pipeline end-to-end in ~a minute
python run_experiments.py --smoke

# the configuration used for the committed figures (3 seeds x 300 iters,
# curves are converged at this horizon)
python run_experiments.py --iters 300 --seeds 0 43 51

# paper-faithful settings (5 seeds x 1000 iters; much longer to run)
python run_experiments.py --iters 1000 --seeds 0 43 51 79 101
```

The figures in `results/` were generated with `--iters 300 --seeds 0 43 51`
on CPU. The reward curves converge well before 300 iterations, so the trends
match the paper; using more seeds/iterations tightens the confidence bands.

Outputs land in `results/`:
- `rewards_<algo>.csv` — mean ± std average-reward curve per algorithm
- `arms_<algo>.csv` — weight-selection distribution (finite bandits)
- `summary.csv` — best weight, best param and final reward per algorithm
- `fig_contextfree_rewards.png` — context-free comparison (paper Fig 2)
- `fig_contextual_rewards.png` — LinUCB vs LinTS (paper Fig 3)
- `fig_regret.png` — cumulative-regret companion curves
- `fig_arms_<algo>.png` — corrected weight-selection distributions

## Environment & base skills

`gym_repoman/` is the pixel gridworld (`CollectEnv`): the agent moves N/E/S/W,
picks up coloured shapes, and observes an 84×84 RGB image. The two base skills
are Deep Q-Networks pre-trained to collect a single object type; they live under
`models/purple_circle/` and `models/beige_crate/` and are blended at run time by
`dqn.ComposedDQN`. `map.png` shows the layout.

## Repository layout

```
run_experiments.py      # single reproducible driver (all algorithms -> CSVs + figures)
composition_reward.py   # shared reward: roll out the weighted composition, return reward
dqn.py                  # DQN + ComposedDQN (weighted value-function composition)
trainer.py              # base-skill training / loading
gym_repoman/            # CollectEnv pixel gridworld (gym env)
wrappers.py             # observation / episode wrappers

# Bandit algorithms (canonical, one per method):
Softmax.py  UCB.py  EXP3.py              # finite
hoo.py                                   # continuous: HOO
lghoo.py                                 #   LG-HOO = memory/compute-efficient HOO implementation
Run_zoom.py  algorithms.py               # continuous: Zooming (impl vendored in algorithms.py)
LinUCB_Final.py  LinTS.py                # contextual
```

## Notes on this release

This is a cleaned, reproducible version of the original research code:
- A single shared `composed_reward` replaces several inconsistent, duplicated
  reward calls; the reward computation itself is unchanged from the paper.
- One `run_experiments.py` driver replaces scattered per-file `__main__` blocks
  and manual results-file stitching.
- Ported to modern `gym` (0.26) — the gridworld now uses the 5-tuple `step` /
  `(obs, info)` `reset` API.
- Duplicate algorithm variants (e.g. `LinUCB`, `LinUCB2`, `LinUCB3`) were
  removed; the canonical implementation of each method is kept at top level.

The contextual experiments use a randomly-generated synthetic context exactly as
described in the paper (Section 5.3); this is intentional and preserved.

## How to cite

This repository is the companion code for:

> *Weighted composition for entropy-regularised reinforcement learning.*
> ORiON (Journal of the Operations Research Society of South Africa).
> DOI: [10.5784/41-2-784](https://doi.org/10.5784/41-2-784) ·
> Article: https://orion.journals.ac.za/pub/article/view/784

BibTeX (fill in the author list / year from the published version):

```bibtex
@article{nyabadza2026weighted,
  title   = {Weighted composition for entropy-regularised reinforcement learning},
  author  = {Nyabadza, Caston and Rosman, Benjamin and James, Steven and Nangue Tasse, Geraud},
  journal = {ORiON},
  volume  = {41},
  number  = {2},
  year    = {2026},
  doi     = {10.5784/41-2-784},
  url     = {https://orion.journals.ac.za/pub/article/view/784}
}
```

If you use this code, please cite the paper above.

## Credits

The weighted-composition method builds on van Niekerk et al., *Will it blend?
Composing value functions in reinforcement learning* (2018), and the broader
bandit literature cited in the paper. This repository is the experimental code
accompanying the ORiON article.
