import numpy as np

def BernoulliArm(p):
    return np.random.choice(2, 1, p=[1-p, p])[0]