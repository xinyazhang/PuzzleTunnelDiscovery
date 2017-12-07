import pyosr
import numpy as np
import random
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

def random_unit_vector(size):
    mag = 0.0
    while mag == 0.0:
        vec = np.random.uniform(size=size)
        mag = np.linalg.norm(vec)
    return vec / mag

def random_continuous_action_2(max_stepping):
    stepping = max_stepping * random.random()
    ratio = random.random()
    tmag = stepping * (1-ratio)
    rmag = stepping * ratio
    tpart = (random_unit_vector(size=(3)) - 0.5) * tmag * 2
    rpart = (random_unit_vector(size=(3)) - 0.5) * rmag * 2
    # print('tpart {}'.format(tpart))
    return tpart, rpart

def random_continuous_action(max_stepping):
    tpart, rpart = random_continuous_action_2(max_stepping)
    return np.concatenate((tpart, rpart))

def random_path(uw, max_stepping, node_num):
    state = uw.translate_to_unit_state(gen_init_state(uw))
    keys = [state]
    ratio = 0.0
    actions = []
    for i in range(node_num - 1):
        done = False
        while not done:
            if ratio < 1.0:
                '''
                Only re-generate direction after hitting things
                '''
                tpart, rpart = random_continuous_action_2(max_stepping)
            nstate, done, ratio = uw.transit_state_by(keys[-1],
                    tpart,
                    rpart,
                    max_stepping / 32)
        # print(tpart, rpart, ratio)
        keys.append(nstate)
        # print(np.concatenate((tpart, rpart)))
        # actions.append(np.concatenate((tpart, rpart)))
        actions.append(pyosr.differential(keys[-2], keys[-1]))
        # print(pyosr.differential(keys[-2], keys[-1]))
    return keys, actions

