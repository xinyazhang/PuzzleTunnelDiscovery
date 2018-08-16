#!/usr/bin/env python2

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

def tunnel_segment_finder(gtfn, pathfn, outfn):
    r = _create_r()
    gtdic = np.load(gtfn)
    pathdic = np.load(pathfn)
    V0 = gtdic['V'][:4]
    V1 = pathdic['VS'][:2]
    VM=r.calculate_visibility_matrix2(V0, False,
                                      V1, False,
                                      0.0125 * 4 / 8)
    np.savez(outfn, PATHVM=VM)

def usage():
    print('''
Find least visible samples
    ''')
    print('''
Usage: tunnel-finder.py <npz file from rl-precalcmap.py> <npz file from visibility-filder> <output tunnel sample npz>
'''
    )

if __name__ == '__main__':
    #tunnel_finder(sys.argv[1], sys.argv[2], sys.argv[3])
    tunnel_segment_finder('blend-low.gt.npz',
                          '../res/alpha/alpha-1.2.org.w-first.npz',
                          'blend-low.path-visibility.npz')
