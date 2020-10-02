# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import pyosr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import aniconf12 as aniconf
import sys

def interpolate(pkey, nkey, tau):
    return pyosr.interpolate(pkey, nkey, tau)

def verifymap(mapfn):
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
    '''
    st0 = np.array([21.782108575648873,11.070742691783639,13.072090341969885,0.99496368307688909,-0.050573680994590003,0.08255004745739393,0.025981951687884433], dtype=np.float64)
    st0 = r.translate_to_unit_state(st0.transpose())
    st1 = np.array([24.404447383193428,16.614281021136808,17.241012748680941,0.89856334412742611,-0.42392368380753659,0.035352511370216902,0.10780921499298371], dtype=np.float64)
    st1 = r.translate_to_unit_state(st1.transpose())
    print(r.transit_state_to(st0, st1, 0.00125))
    return
    '''
    gt = pyosr.GTGenerator(r)
    gt.rl_stepping_size = 0.0125
    # gt.verify_magnitude = 0.0125
    gt.load_roadmap_file(mapfn)
    # gt.init_gt()
    gt.save_verified_roadmap_file(mapfn + ".verified")
    return

def usage():
    print("rl-verifymap.py <blended roadmap file>")

if __name__ == '__main__':
    '''
    verifymap('')
    exit()
    '''
    if len(sys.argv) < 2:
        usage()
        exit()
    verifymap(sys.argv[1])
