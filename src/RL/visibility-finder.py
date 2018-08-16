#!/usr/bin/env python2

import os
import sys
sys.path.append(os.getcwd())

import pyosr
import numpy as np
import math
import aniconf12 as aniconf
import sys
import uw_random

def visibility(gtfn, cfn):
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
    d = np.load(gtfn)
    V=d['V']
    VM=r.calculate_visibility_matrix(V, False, 0.0125 * 4 / 8)
    np.savez(cfn, VM=VM)

def usage():
    print('''
Calculate the visibility matrix of given input files.
    ''')
    print("Usage: visibility-finder.py <npz file outputed from rl-precalcmap.py> <output npz file>")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        exit()
    visibility(sys.argv[1], sys.argv[2])
