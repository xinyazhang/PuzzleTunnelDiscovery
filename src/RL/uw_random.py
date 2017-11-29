import pyosr
import numpy as np
from math import sqrt,pi,sin,cos

def random_state(scale=1.0):
    tr = scale * (np.random.rand(3) - 0.5)
    u1,u2,u3 = np.random.rand(3)
    quat = [sqrt(1-u1)*sin(2*pi*u2),
            sqrt(1-u1)*cos(2*pi*u2),
            sqrt(u1)*sin(2*pi*u3),
            sqrt(u1)*cos(2*pi*u3)]
    part1 = np.array(tr, dtype=np.float32)
    part2 = np.array(quat, dtype=np.float32)
    return np.concatenate((part1, part2))

def gen_init_state(uw):
    while True:
        state = random_state()
        if uw.is_disentangled(state):
            continue
        if uw.is_valid_state(state):
            break
    return uw.translate_from_unit_state(state)

def random_continuous_action(max_stepping):
    stepping = max_stepping * random.random()
    rratio = random.random()
    tmag = stepping * (1-ratio)
    rmag = stepping * ratio
    tpart = np.linalg.norm(np.random.uniform(shape=(3))) * tmag
    rpart = np.linalg.norm(np.random.uniform(shape=(3))) * rmag
    return np.concatenate((tpart, rpart))

