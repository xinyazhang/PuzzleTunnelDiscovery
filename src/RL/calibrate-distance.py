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
    pos = []
    posd = []
    neg = []
    negd = []
    for i in range(len(V)):
        V[i] = r.translate_to_unit_state(V[i])
        if r.is_disentangled(V[i]):
            D[i] = 0.0
    # Trim distant samples, which usually shows nothing
    for i in range(len(V)):
        if pyosr.distance(origin, V[i]) > 0.75:
            continue
        pos.append(V[i])
        posd.append(D[i])
    V = np.array(pos)
    D = np.array(posd)
    for i in range(len(V)):
        while True:
            s = uw_random.random_state()
            if not r.is_disentangled(s):
                break
        neg.append(s)
        negd.append(-10.0)
    NV = np.array(neg)
    ND = np.array(negd)
    V = np.concatenate((V, NV))
    D = np.concatenate((D, ND))
    p = np.random.permutation(len(V))
    np.savez(cfn, V=V[p], E=d['E'], D=D[p], N=d['N'])

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
