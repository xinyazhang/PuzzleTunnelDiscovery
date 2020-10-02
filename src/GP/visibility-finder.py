#!/usr/bin/env python2
# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later

import os
import sys
sys.path.append(os.getcwd())

import pyosr
import numpy as np
import math
import aniconf12 as aniconf

def _create_r():
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
    return r

def visibility(gtfn, cfn):
    r = _create_r()
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
