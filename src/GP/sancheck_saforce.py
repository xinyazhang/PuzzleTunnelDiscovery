#!/usr/bin/env python2
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

import os
import sys
sys.path.append(os.getcwd())

import pyosr
import phyutil
import numpy as np
import math
import aniconf12 as aniconf
import rlreanimator

def _create_r():
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    r = pyosr.Renderer()
    r.setup()
    r.loadModelFromFile(aniconf.env_wt_fn)
    r.loadRobotFromFile(aniconf.rob_wt_fn)
    r.enforceRobotCenter(aniconf.rob_ompl_center)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    r.views = np.array([[0.0, 0.0]], dtype=np.float32)
    r.avi = True
    return r

def main():
    r = _create_r()
    q = np.array([0.19084715098142624, 0.41685351729393005, 0.23655708134174347, 0.9332677282828477, -0.18945885280431052, 0.20543615363339196, -0.2256383770996524])
    tup = r.intersecting_segments(q)
    print(tup[2])
    fposs,fdirs = r.force_direction_from_intersecting_segments(tup[0], tup[1], tup[3])
    print(fposs)
    print(fdirs)
    env_normals = r.scene_face_normals_from_index_pairs(tup[3])
    wrong_direction = False
    for fdir,en in zip(fdirs, env_normals):
        if np.dot(fdir, en) < 0.0:
            print("San check 1: force direction w.r.t. face normal failed.")
            wrong_direction = True
            break
    if not wrong_direction:
        print("San check 1: force direction w.r.t. face normal passed.")
    def imager(q):
        QS = []
        for q in phyutil.collision_resolve(r, q):
            QS.append(q)
            r.state = q
            r.render_mvrgbd()
            rgb = np.copy(r.mvrgb.reshape((r.pbufferWidth, r.pbufferHeight, 3)))
            yield rgb # First view
            print('state {}'.format(q))
        QS.append(q)
        print("Total: {} steps".format(len(QS)))
        np.savez('saforce_process.npz', QS=QS)
    rlreanimator.reanimate(imager(q), fps=20)

"""
def usage():
    print('''
Calculate the visibility matrix of given input files.
    ''')
    print("Usage: visibility-finder.py <npz file outputed from rl-precalcmap.py> <output npz file>")
"""

if __name__ == '__main__':
    main()
