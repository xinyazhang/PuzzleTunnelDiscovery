'''
RL Continuous Action
    - Input: output file of calibrate-distance.py
'''

import pyosr
import numpy as np

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
    NV = len(V)
    while True:
        ivi = np.random.randint(NV)
        uv = uw.translate_to_unit_state(V[ivi])
        if uw.is_valid_state(uv) and not uw.is_disentangled(uv) and not N[ivi] < 0:
            break
        '''
        if N[ivi] >= 0:
            break
        '''
    print("IVI {}".format(ivi))
    # print("N[8050] {}".format(N[8050]))
    # ivi = 8050
    #ivi = 0
    cvi = ivi
    vs = []
    while True:
        print('CHECKING {} {}'.format(cvi, V[cvi]))
        uv = uw.translate_to_unit_state(V[cvi])
        if vs and pyosr.distance(uv, vs[-1]) < amag / 10.0:
            pass # Do not add extremly close vertices
        else:
            vs.append(uv)
        if uw.is_disentangled(vs[-1]):
            break
        cvi = N[cvi]
        if cvi < 0:
            break
    print(vs)
    path = np.array(vs)
    for i,(cv,nv) in enumerate(zip(path[:-1], path[1:])):
        mag = pyosr.distance(cv, nv)
        tr, aa = pyosr.differential(cv, nv)
        applied = pyosr.apply(cv, tr, aa)
        '''
        print("mag {}".format(mag))
        print("cv {}".format(cv))
        print("nv {}".format(nv))
        print("tr {}".format(tr))
        print("aa {}".format(aa))
        print("applied {}".format(applied))
        print("applied quat norm {}".format(np.linalg.norm(applied[3:])))
        '''
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
        last_tr, last_aa = pyosr.differential(cc, nv)
        yield cc, last_tr, last_aa

