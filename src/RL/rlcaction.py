# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
'''
RL Continuous Action
    - Input: output file of calibrate-distance.py
'''

import pyosr
import numpy as np

def ivi_to_leaving_trajectory(ivi, V, N, D, amag, uw):
    # print("N[8050] {}".format(N[8050]))
    # ivi = 20146
    # ivi = 0
    cvi = ivi
    vs = []
    while True:
        # print('CHECKING {} {}'.format(cvi, V[cvi]))
        uv = uw.translate_to_unit_state(V[cvi])
        if vs and pyosr.distance(uv, vs[-1]) < amag / 10.0:
            pass # Do not add extremly close vertices
        else:
            #if uv[3] < 0.0:
            #    uv[3:] *= -1.0
            vs.append(uv)
        vs.append(uv)
        if uw.is_disentangled(vs[-1]):
            break
        cvi = N[cvi]
        if cvi < 0:
            break
    if uw.is_disentangled(vs[-1]):
        return vs
    else:
        return []

def sample_a_leaving_trajectory(V, N, D, amag, uw):
    NV = len(V)
    vs = []
    while not vs:
        while True:
            ivi = np.random.randint(NV)
            # ivi = 20146
            uv = uw.translate_to_unit_state(V[ivi])
            if uw.is_valid_state(uv) and not uw.is_disentangled(uv) and not N[ivi] < 0:
                break
            '''
            if N[ivi] >= 0:
                break
            '''
        #ivi = 242
        print("IVI {}".format(ivi))
        assert uw.is_valid_state(uv)
        vs = ivi_to_leaving_trajectory(ivi, V, N, D, amag, uw)
    return vs

'''
Translate a list of vertices into a sequence of descrete actions
    vs: list of vertices
    uw: UnitWorld object
'''
def trajectory_to_caction(vs, uw, amag):
    path = np.array(vs)
    for i,(cv,nv) in enumerate(zip(path[:-1], path[1:])):
        assert uw.is_valid_state(cv), "{} is not valid state".format(cv)
        mag = pyosr.distance(cv, nv)
        tr, aa, dquat = pyosr.differential(cv, nv)
        applied = pyosr.apply(cv, tr, aa)
        if True:
            print("mag {}".format(mag))
            print("cv {}".format(cv))
            print("nv {}".format(nv))
            print("tr {}".format(tr))
            print("aa {}".format(aa))
            print("dquat {}".format(dquat))
            print("applied {}".format(applied))
            print("applied quat norm {}".format(np.linalg.norm(applied[3:])))
        assert pyosr.distance(applied, nv) < 1e-6, "pyosr.apply is not the inverse of pyosr.differential {} != {}".format(applied, nv)
        action_step = 0
        stepping_tr = tr / mag * amag
        stepping_aa = aa / mag * amag
        cc = np.copy(cv)
        while pyosr.distance(cc, nv) > amag:
            yield cc, stepping_tr, stepping_aa
            if uw.is_disentangled(cc):
                return
            cc = pyosr.apply(cc, stepping_tr, stepping_aa)
        last_tr, last_aa, dquat = pyosr.differential(cc, nv)
        yield cc, last_tr, last_aa


'''
Generator that yields (state, caction translation, caction angle axis) tuples
Input
    - envir: rlenv.IEnvironment
    - V: (NV, 7) matrix, vertices
    - N: (NV) matrix, next vertices
    - D: (ND) matrix, distance to goal state
    - amag: float, action magnititude
'''
def caction_generator(V, N, D, amag, uw):
    vs = sample_a_leaving_trajectory(V, N, D, amag, uw)
    # assert uw.is_disentangled(vs[-1])
    print(vs)
    for items in trajectory_to_caction(vs, uw, amag):
        yield items

import uw_random

def _gen_unit_init_state(uw, origin, radius):
    while True:
        state = uw_random.random_state(radius)
        state[0:3] += origin[0:3]
        if uw.is_disentangled(state):
            continue
        if uw.is_valid_state(state):
            return state

def caction_generator2(uw, known_path, is_radius, amag):
    assert isinstance(known_path, list), 'known_path should be a list rather than something else like np.array'
    print("### caction_generator2: FINDING INITIAL STATE")
    DEBUG=False
    if DEBUG:
        ni = 50
        iv = known_path[ni - 1]
    else:
        while True:
            # iv = uw_random.gen_unit_init_state(uw, is_radius)
            iv = _gen_unit_init_state(uw, known_path[0], is_radius)
            distances = pyosr.multi_distance(iv, known_path)
            ni = np.argmin(distances)
            # ni = 0
            '''
            if uw.is_valid_transition(iv, known_path[ni], amag / 8.0):
                break
            '''
            fstate, done, ratio = uw.transit_state_to(known_path[ni], iv, amag / 8.0)
            if ratio > 1e-3:
                iv = fstate
                break
    vs = [iv] + known_path[ni:]
    # print('VS\n{}'.format(vs))
    # vs = known_path
    for items in trajectory_to_caction(vs, uw, amag):
        yield items
