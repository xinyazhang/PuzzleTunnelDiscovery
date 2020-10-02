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

def reanimate(gtfn, anifn):
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
    # gt.rl_stepping_size = 0.0125 / 64 # it's for animation only if AA is enabled
    gt.verify_magnitude = 0.0125 / 64 / 8
    # print('default rl_stepping_size {}'.format(gt.rl_stepping_size))
    gtdata = np.load(gtfn)
    gt.install_gtdata(gtdata['V'], gtdata['E'], gtdata['D'], gtdata['N'])
    # FIXME: add this back after debugging Dijkstra
    # gt.init_knn_in_batch()

    print(r.scene_matrix)
    print(r.robot_matrix)
    # return

    class ReAnimator(object):
        reaching_terminal = False
        driver = None
        im = None
        keys = None
        prev_state = None

        def __init__(self, renderer, keys, delta=0.125):
            self.renderer = renderer
            self.keys = keys
            self.t = 0.0
            self.delta = delta
            self.reaching_terminal = False
            self.prev_state = self.renderer.translate_to_unit_state(keys[0])

        def perform(self, framedata):
            r = self.renderer
            index = int(math.floor(self.t))
            nindex = index + 1
            if nindex < len(self.keys) and not self.reaching_terminal:
                pkey = self.keys[index]
                nkey = self.keys[nindex]
                tau = self.t - index
                print("tau {}".format(tau))
                state = interpolate(pkey, nkey, tau)
                r.state = r.translate_to_unit_state(state)
                r.render_mvrgbd()
                # print(r.mvrgb.shape)
                rgb = r.mvrgb.reshape((r.pbufferWidth, r.pbufferHeight, 3))
                # print(r.state)
                valid = r.is_valid_state(r.state)
                print('\tNew State {} Collision Free ? {}'.format(r.state, valid))
                print('\tDistance {}'.format(pyosr.distance(r.state, self.prev_state)))
                if not valid:
                    print('\tNOT COLLISION FREE, SAN CHECK FAILED AT {}'.format(self.t))
                    self.reaching_terminal = True
                if self.im is None:
                    print('rgb {}'.format(rgb.shape))
                    self.im = plt.imshow(rgb)
                else:
                    self.im.set_array(rgb)
                self.prev_state = r.state
                self.t += self.delta

    fig = plt.figure()
    keys = np.loadtxt(aniconf.keys_fn)
    keys[:, [3,4,5,6]] = keys[:,[6,3,4,5]]
    init_state = keys[0]
    # init_state = np.array([0.36937377,0.1908864,0.21092376, 0.98570921, -0.12078825, -0.0675864, 0.09601888], dtype=np.float64)
    # gt.init_knn_in_batch()

    if anifn is None:
        STEP_LIMIT = 64 * 1024
        FOR_RL = False
        gtstats, actions, terminated = gt.generate_gt_path(init_state, STEP_LIMIT, FOR_RL)
        if STEP_LIMIT >= 1024:
            np.savez('data-from-init_state', S=gtstats, A=actions)
            return
        print("terminated ? {} Expecting True".format(terminated))
    else:
        '''
        gtdata = np.load(anifn)
        gtstats = gtdata['S']
        '''
        gtstats = np.loadtxt(anifn)
        # gtstats[:, [3,4,5,6]] = gtstats[:,[6,3,4,5]]
        print(gtstats.shape)
    # print('trajectory:\n')
    # print(gtstats)
    ra = ReAnimator(r, gtstats, 0.05)
    ani = animation.FuncAnimation(fig, ra.perform)
    plt.show()

def usage():
    print("rlaa-reanimator.py <preprocessed ground truth file>")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        exit()
    if len(sys.argv) >= 3:
        anifn = sys.argv[2]
    else:
        anifn = None
    reanimate(sys.argv[1], anifn)
