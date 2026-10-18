"""
composition_reward.py — shared reward for the weighted-composition bandit.

This is the single, canonical reward used by every bandit algorithm in this
repository (Softmax, UCB, EXP3, LinUCB, LinTS, HOO). It replaces the
duplicated/inconsistent `MaxLength.reward(...)` calls that previously lived in
experiment_weighted_or.py and were invoked inconsistently across files (some
abusing the constructor as `MaxLength(gym.Wrapper, 50)`, some unpacking a tuple
the function never returned).

The REWARD COMPUTATION is unchanged from the published experiments: compose the
two entropy-regularised base skills (beige_crate, purple_circle) with weights
[w, 1-w], roll out one episode of up to `max_trajectory` steps, and return a
reward linked to (a) the number of steps taken to reach the target and (b) which
object was collected — exactly as described in the paper (Section 5).

Return contract (now consistent everywhere):
    composed_reward(max_trajectory, weight) -> (reward: float, collected: str|None)

`collected` is 'BS' (beige square), 'PC' (purple circle) or None (nothing
collected within the step limit). Callers that only need the scalar use [0].
"""

import os
# Headless pygame: the CollectEnv renders to an off-screen surface, so no
# display is needed. Must be set before pygame is imported (via CollectEnv).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gym
import torch
import numpy as np

from dqn import ComposedDQN, FloatTensor, get_action
from trainer import load
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class _MaxLength(gym.Wrapper):
    """Episode-length cap wrapper (ends the episode after `length` steps).

    Uses the modern gym (>=0.26) 5-tuple step / (obs, info) reset API.
    """

    def __init__(self, env, length):
        gym.Wrapper.__init__(self, env)
        self.max_length = length
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        if self.steps == self.max_length:
            truncated = True
        return ob, reward, terminated, truncated, info


# Base skills are loaded once and reused across all bandit pulls (loading a DQN
# per pull was both slow and a source of per-call state). Lazy-initialised so
# importing this module has no side effects.
_BASE_SKILLS = None


def _get_base_skills(task):
    """Load the two entropy-regularised base-skill DQNs once, cached."""
    global _BASE_SKILLS
    if _BASE_SKILLS is None:
        dqn_purple_circle = load('./models/purple_circle/model.dqn', task)
        dqn_beige_crate = load('./models/beige_crate/model.dqn', task)
        dqn_purple_circle.to(device)
        dqn_beige_crate.to(device)
        dqn_purple_circle.eval()
        dqn_beige_crate.eval()
        _BASE_SKILLS = (dqn_beige_crate, dqn_purple_circle)
    return _BASE_SKILLS


def composed_reward(max_trajectory, weight):
    """Roll out one episode of the weighted composition and return its reward.

    Args:
        max_trajectory: episode step cap.
        weight: composition weight w in [0, 1]; skills are combined as
                [w, 1 - w] over [beige_crate, purple_circle].

    Returns:
        (reward, collected): reward is a float linked to steps-to-target and the
        object collected; collected is 'BS', 'PC', or None.
    """
    env = _MaxLength(
        WarpFrame(CollectEnv(goal_condition=lambda x: (x.colour == 'beige' and x.shape == 'square')
                             or (x.colour == 'purple' and x.shape == 'circle'))),
        max_trajectory)

    dqn_beige_crate, dqn_purple_circle = _get_base_skills(env)
    dqn_composed = ComposedDQN([dqn_beige_crate, dqn_purple_circle], [weight, 1 - weight])

    obs, _ = env.reset()
    steps = 0
    for _ in range(max_trajectory):
        obs = torch.from_numpy(np.array(obs)).type(FloatTensor).unsqueeze(0)
        action = get_action(dqn_composed, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            collected = info['collected']
            if len([c for c in collected if c.colour == 'beige' and c.shape == 'square']) > 0:
                return -1.0 * steps + np.random.normal(0, 1), 'BS'
            elif len([c for c in collected if c.colour == 'purple' and c.shape == 'circle']) > 0:
                return -1.0 * steps + np.random.random_sample(), 'PC'
            else:
                return -1.0 * steps - 2.0, None
    # step cap reached without collecting anything
    return -1.0 * max_trajectory - 2.0, None
