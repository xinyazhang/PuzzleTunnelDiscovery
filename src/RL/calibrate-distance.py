import pyosr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import aniconf12 as aniconf
import sys
import uw_random

def calibrate(gtfn, cfn):
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    r = pyosr.Renderer()
    r.setup()
    r.loadModelFromFile(aniconf.env_fn)
    r.loadRobotFromFile(aniconf.rob_fn)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    r.views = np.array([[0.0, 0.0]], dtype=np.float32)
    origin = np.array([0,0,0,1,0,0,0], dtype=np.float64)
    d = np.load(gtfn)
    V=d['V']
    D=d['D']
    N=d['N']
    pos = []
    posd = []
    neg = []
    negd = []
    for i in range(len(V)):
        V[i] = r.translate_to_unit_state(V[i])
        if r.is_disentangled(V[i]):
            D[i] = 0.0
    assert len(d['N']) == len(V), "N size ({}) does not match V's ({})".format(len(d['N']), len(V))
    print("istate {}  distance {}".format(V[0], pyosr.distance(origin, V[0])))
    print("gstate {}  distance {}".format(V[1], pyosr.distance(origin, V[1])))
    # Trim distant samples, which usually shows nothing
    old_to_new = {}
    new_to_old = {}
    NN = []
    for i in range(len(V)):
        if pyosr.distance(origin, V[i]) > 0.85:
            continue
        old_to_new[i] = len(pos)
        pos.append(V[i])
        posd.append(D[i])
    for i in range(len(V)):
        if pyosr.distance(origin, V[i]) > 0.85:
            continue
        if N[i] < 0:
            NN.append(N[i])
        else:
            NN.append(old_to_new[N[i]])
    NE = []
    for e in d['E']:
        if e[0] not in old_to_new or e[1] not in old_to_new:
            continue
        NE.append([old_to_new[e[0]], old_to_new[e[1]]])
    assert len(NN) == len(pos), "NN size ({}) does not match pos' ({})".format(len(NN), len(pos))
    V = np.array(pos)
    D = np.array(posd)
    for i in range(len(V)):
        while True:
            s = uw_random.random_state()
            if not r.is_disentangled(s):
                break
        neg.append(s)
        negd.append(-10.0)
        NN.append(-1)
    NV = np.array(neg)
    ND = np.array(negd)
    V = np.concatenate((V, NV))
    D = np.concatenate((D, ND))
    E = np.array(NE)
    N = np.array(NN)
    #p = np.random.permutation(len(V))
    #np.savez(cfn, V=V[p], E=E, D=D[p], N=N[p])
    np.savez(cfn, V=V, E=E, D=D, N=N)

def usage():
    print('''
Prepare the ground truth samples for Q learning
1. calibrate D to zero if it is solved state
2. Add the same number of negative samples
    ''')
    print("Usage: calibrate-distance.py  <npz file outputed from rl-precalcmap.py> <output npz file>")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        exit()
    calibrate(sys.argv[1], sys.argv[2])
