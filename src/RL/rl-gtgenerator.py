# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import pyosr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import aniconf12 as aniconf
import sys
import config

MAX_GT_STATES = 4 * 1024 # log(0.01)/log(0.999) ~= 4602 ~= 4096

def interpolate(pkey, nkey, tau):
    return pyosr.interpolate(pkey, nkey, tau)

def gtgen(gtfn, gtdir, gtnumber):
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
    gt = pyosr.GTGenerator(r)
    gt.rl_stepping_size = 0.0125
    gtdata = np.load(gtfn)
    gt.install_gtdata(gtdata['V'], gtdata['E'], gtdata['D'], gtdata['N'])
    gt.init_knn_in_batch()

    print(r.scene_matrix)
    print(r.robot_matrix)
    # return

    for i in range(gtnumber):
        # Random state, copied from RLDriver.restart_epoch
        while True:
            tr = np.random.rand(3) * 2.0 - 1.0
            u1,u2,u3 = np.random.rand(3)
            quat = [sqrt(1-u1)*sin(2*pi*u2),
                    sqrt(1-u1)*cos(2*pi*u2),
                    sqrt(u1)*sin(2*pi*u3),
                    sqrt(u1)*cos(2*pi*u3)]
            part1 = np.array(tr, dtype=np.float32)
            part2 = np.array(quat, dtype=np.float32)
            r.state = np.concatenate((part1, part2))
            if r.is_disentangled(r.state):
                continue
            if r.is_valid_state(r.state):
                break
        gtstats, gtactions, terminated = gt.generate_gt_path(r.state, MAX_GT_STATES)
        if not terminated:
            '''
            Try again
            '''
            continue

def usage():
    print("rl-gtgenerator.py <preprocessed ground truth file> <ground truth directory> <number of paths>")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        usage()
        exit()
    gtgen(sys.argv[1], sys.argv[2], int(sys.argv[3]))
