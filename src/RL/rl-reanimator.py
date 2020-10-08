# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
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
    gt.rl_stepping_size = 0.0125 / 64
    gt.verify_magnitude = 0.0125 / 64 / 8
    '''
    # gt.rl_stepping_size = 0.0125 / 16
    # gt.verify_magnitude = 0.0125 / 16 / 8
    #r.state = np.array([0.36937377,0.1908864,0.21092376, 0.98570921, -0.12078825, -0.0675864, 0.09601888], dtype=np.float64)
    # r.state = np.array([0.36937377,0.1908864,0.21092376, 1, 0, 0, 0], dtype=np.float64)
    # from_state = np.array([24.896620637662053,41.988471340054012,20.017354233307341,0.6418730497277253,-0.44418703270561571,0.59576996921466863,-0.18908995687598346], dtype=np.float64)
    # from_state = np.array([21.774434817350986,20.121317084495768,18.484999518340139,0.86627823035212614,-0.41451860285720288,0.23441339549279344,0.15095269297644118], dtype=np.float64)
    from_state = np.array([27.29668045703666124, 19.51906472394184533, 16.65767570144958398, 0.87262677298191904,-0.41840872052286393,0.21772088243415724,0.12670546559940116], dtype=np.float64)
    r.state = r.translate_to_unit_state(from_state)
    # to_state = np.array([24.934771980787307,42.024757337036505,20.013917547396996,0.64195016768344282,-0.44418881752482781,0.5956043578857666,-0.18934551873381589], dtype=np.float64)
    to_state = np.array([21.624491470438645,20.262838222930512,18.597999780952151,0.87266847945437775,-0.40534409067552768,0.23710605772313775,0.13389029282291187], dtype=np.float64)
    next_state = r.translate_to_unit_state(to_state)
    print("Projection {}".format(gt.project_trajectory(from_state, to_state)))
    print("Valid init state? {}".format(r.is_valid_state(r.state)))
    print("Distance from start to end? {}".format(pyosr.distance(r.state, next_state)))
    print("Stepping/Verify Mag? {}".format([gt.rl_stepping_size, gt.verify_magnitude]))
    mag = pyosr.distance(r.state, next_state)
    for a in range(12):
        tup = r.transit_state(r.state, a, gt.rl_stepping_size, gt.verify_magnitude)
        print(tup)
        print("distance from start: {}".format(pyosr.distance(tup[0], r.state)))
        d = pyosr.distance(tup[0], next_state)
        print("distance to next: {} Closer? {}".format(d, d < mag))
    return
    '''
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

    '''
    gtstats, actions, terminated = gt.generate_gt_path(init_state, 64 * 1024, False)
    np.savetxt('blend.path', gtstats)
    return
    '''

    if anifn is None:
        STEP_LIMIT = 64 * 1024
        FOR_RL = True
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
    print("rl-reanimator.py <preprocessed ground truth file>")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        exit()
    if len(sys.argv) >= 3:
        anifn = sys.argv[2]
    else:
        anifn = None
    reanimate(sys.argv[1], anifn)
