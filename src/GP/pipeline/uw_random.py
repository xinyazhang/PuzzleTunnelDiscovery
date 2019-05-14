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

def random_on_sphere(scale=1.0):
    '''
    x1,x2 in (-1,1)
    '''
    while True:
        x1,x2 = 2.0 * (np.random.rand(2) - 0.5)
        l2 = x1 * x1 + x2 * x2
        if l2 < 1:
            break
    '''
    Formula 9-11 in http://mathworld.wolfram.com/SpherePointPicking.html
    '''
    x = 2 * x1 * sqrt(1 - l2)
    y = 2 * x2 * sqrt(1 - l2)
    z = 1 - 2 * l2
    return np.array([x,y,z]) * scale

def random_within_sphere(scale=1.0):
    # sample within [-1,1]^3
    # and reject samples not in the unit sphere
    while True:
        x1,x2,x3 = 2.0 * (np.random.rand(3) - 0.5)
        l2 = x1 * x1 + x2 * x2 + x3 * x3
        if l2 <= 1:
            break
    return np.array([x1,x2,x3]) * scale

def gen_unit_init_state(uw, scale=1.0):
    while True:
        state = random_state(scale)
        if uw.is_disentangled(state):
            continue
        if uw.is_valid_state(state):
            return state

def gen_init_state(uw):
    return uw.translate_from_unit_state(gen_unit_init_state(uw))

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

def random_path(uw, max_stepping, node_num, scale=0.5):
    state = gen_unit_init_state(uw, scale)
    keys = [state]
    ratio = 0.0
    actions = []
    '''
    Lazy import
    '''
    import pyosr
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
        '''
        Actual T,R
        '''
        tpart, rpart = pyosr.differential(keys[-2], keys[-1])
        actions.append(np.concatenate((tpart, rpart)))
        # print(pyosr.differential(keys[-2], keys[-1]))
    return keys, actions

DISCRETE_ACTION_NUMBER = 12

def random_discrete_path_v0(uw, action_magnitude, verify_magnitude, node_num):
    state = uw.translate_to_unit_state(gen_init_state(uw))
    node_num = 2
    keys = [state]
    ratio = 0.0
    actions = []
    banned_actions = np.zeros(DISCRETE_ACTION_NUMBER, dtype=np.int32)
    for i in range(node_num - 1):
        done = False
        action = random.randrange(2)
        nstate, done, ratio = uw.transit_state(keys[-1],
                action,
                action_magnitude,
                verify_magnitude)
        # print(tpart, rpart, ratio)
        keys.append(nstate)
        # print('> VERBOSE: action {} next state: {} ratio: {}'.format(action, nstate, ratio))
        # print(np.concatenate((tpart, rpart)))
        actions.append(action)
        # print(pyosr.differential(keys[-2], keys[-1]))
    return keys, actions

def random_discrete_path_v1(uw, action_magnitude, verify_magnitude, node_num):
    state = uw.translate_to_unit_state(gen_init_state(uw))
    keys = [state]
    ratio = 0.0
    actions = []
    banned_actions = np.zeros(DISCRETE_ACTION_NUMBER, dtype=np.int32)
    for i in range(node_num - 1):
        done = False
        while True:
            if ratio < 1.0:
                '''
                Only re-generate direction after hitting things

                and avoid banned actions
                '''
                while True:
                    action = random.randrange(DISCRETE_ACTION_NUMBER)
                    if banned_actions[action] == 0:
                        break;
            else:
                '''
                Enable all actions
                '''
                banned_actions = np.zeros(DISCRETE_ACTION_NUMBER, dtype=np.int32)
            nstate, done, ratio = uw.transit_state(keys[-1],
                    action,
                    action_magnitude,
                    verify_magnitude)
            if ratio < 1.0:
                '''
                ban current action since cannot move forward.
                '''
                banned_actions[action] = 1
            # print('> Action {} ratio {}'.format(action, ratio))
            if ratio > 0.0:
                break
        # print(tpart, rpart, ratio)
        keys.append(nstate)
        # print('> VERBOSE: action {} next state: {} ratio: {}'.format(action, nstate, ratio))
        # print(np.concatenate((tpart, rpart)))
        actions.append(action)
        # print(pyosr.differential(keys[-2], keys[-1]))
    return keys, actions

def random_discrete_path(uw, action_magnitude, verify_magnitude, node_num):
    state = uw.translate_to_unit_state(gen_init_state(uw))
    keys = [state]
    ratio = 0.0
    actions = []
    banned_actions = np.zeros(DISCRETE_ACTION_NUMBER, dtype=np.int32)
    for i in range(node_num - 1):
        done = False
        '''
        random walking
        '''
        while not done:
            action = random.randrange(2)
            nstate, done, ratio = uw.transit_state(keys[-1],
                    action,
                    action_magnitude,
                    verify_magnitude)
        # print(tpart, rpart, ratio)
        keys.append(nstate)
        # print('> VERBOSE: action {} next state: {} ratio: {}'.format(action, nstate, ratio))
        # print(np.concatenate((tpart, rpart)))
        actions.append(action)
        # print(pyosr.differential(keys[-2], keys[-1]))
    return keys, actions

def random_discrete_path_action_set(uw, action_magnitude, verify_magnitude, node_num, action_set):
    while True:
        state = uw.translate_to_unit_state(gen_init_state(uw))
        keys = [state]
        ratio = 0.0
        actions = []
        success = True
        for i in range(node_num - 1):
            done = False
            action = random.choice(action_set)
            nstate, done, ratio = uw.transit_state(keys[-1],
                    action,
                    action_magnitude,
                    verify_magnitude)
            # print('> VERBOSE: action {} next state: {} ratio: {}'.format(action, nstate, ratio))
            if not done:
                success = False
                break
            # print(tpart, rpart, ratio)
            keys.append(nstate)
            # print(np.concatenate((tpart, rpart)))
            actions.append(action)
            # print(pyosr.differential(keys[-2], keys[-1]))
        if success:
            break
    return keys, actions

