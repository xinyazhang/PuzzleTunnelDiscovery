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
import os.path

kEnableActionCalculation = True

def interpolate(pkey, nkey, tau):
    return pyosr.interpolate(pkey, nkey, tau)

def partition(key0, key1, thresh):
    d = pyosr.distance(key0, key1)
    print("{}\n {}\n\tdist: {}".format(key0, key1, d))
    if d < thresh:
        return []
    mid = interpolate(key0, key1, 0.5)
    print("\t\tmid: {}".format(mid))
    firstlist = partition(key0, mid, thresh)
    secondlist = partition(mid, key1, thresh)
    return firstlist + [mid] + secondlist

def reanimate(gtfn, pathfn, swap, cachedir,interpthresh=0,in_unit=True):
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    r = pyosr.Renderer()
    r.avi = True
    r.setup()
    r.loadModelFromFile(aniconf.env_fn)
    r.loadRobotFromFile(aniconf.rob_fn)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    r.views = np.array([[0.0, 0.0]], dtype=np.float32)
    gt = pyosr.GTGenerator(r)
    # gt.rl_stepping_size = 0.0125 / 16
    # gt.verify_magnitude = 0.0125 / 16 / 8
    # gt.rl_stepping_size = 0.0125 / 16 / 1024
    # gt.verify_magnitude = 0.0125 / 16 / 1024 / 2
    gt.rl_stepping_size = 0.0125 * 4
    gt.verify_magnitude = 0.0125 * 4 / 8
    # gt.rl_stepping_size = 0.0125 / 64
    # gt.verify_magnitude = 0.0125 / 64 / 8
    gtdata = np.load(gtfn)
    gt.install_gtdata(gtdata['V'], gtdata['E'], gtdata['D'], gtdata['N'])

    # FIXME: add this back after debugging Dijkstra
    # gt.init_knn_in_batch()

    class ExpandingReAnimator(object):
        reaching_terminal = False
        driver = None
        im = None
        keys = None
        prev_state = None
        gt = None
        state = None

        def __init__(self, renderer, gt, keys, delta=1):
            self.renderer = renderer
            self.gt = gt
            self.keys = keys
            self.t = 0.0
            self.delta = delta
            self.reaching_terminal = False
            self.prev_state = self.renderer.translate_to_unit_state(keys[0])
            self.st_index = 0
            self.state = self.keys[0]
            self.expand_states()

        def expand_states(self):
            keys = self.keys
            if self.st_index + 1 >= keys.shape[0]:
                self.reaching_terminal = True
                return
            r = self.renderer
            # r.light_position = np.random.rand(3)
            '''
            from_state = r.translate_to_unit_state(keys[self.st_index])
            to_state =  r.translate_to_unit_state(keys[self.st_index + 1])
            print(from_state)
            print(to_state)
            mag = pyosr.distance(from_state, to_state)
            '''
            '''
            for a in range(12):
                tup = r.transit_state(from_state, a, gt.rl_stepping_size, gt.verify_magnitude)
                print(tup)
                print("distance from start: {}".format(pyosr.distance(tup[0], from_state)))
                d = pyosr.distance(tup[0], to_state)
                print("distance to next: {} Closer? {}".format(d, d < mag))
            return
            '''
            fn = '{}/{}.npz'.format(cachedir, self.st_index)
            if not os.path.isfile(fn):
                tup = self.gt.project_trajectory(self.state, keys[self.st_index + 1],in_unit=in_unit)
                self.exp_st = tup[0]
                if self.exp_st.shape[0] > 0:
                    self.state = self.exp_st[-1]
                print("cache states to {}".format(fn))
                np.savez(fn, S=tup[0], A=tup[1])
            else:
                print("load cached states from {}".format(fn))
                cache = np.load(fn)['S']
                self.exp_st = cache
            # print(self.exp_st)
            self.exp_st_index = 0
            self.st_index += 1

        def perform(self, framedata):
            r = self.renderer
            while self.exp_st_index >= self.exp_st.shape[0]:
                self.expand_states()
                if self.reaching_terminal == True:
                    print("Reaching Terminal")
                    return
            '''
            if self.exp_st_index > self.keys.shape[0]:
                return
            r.state = r.translate_to_unit_state(self.keys[self.exp_st_index])
            '''
            r.state = r.translate_to_unit_state(self.exp_st[self.exp_st_index])
            r.render_mvrgbd()
            rgb = r.mvrgb.reshape((r.pbufferWidth, r.pbufferHeight, 3))
            if self.im is None:
                self.im = plt.imshow(rgb)
            else:
                self.im.set_array(rgb)
            self.prev_state = r.state
            self.exp_st_index += self.delta
            print('{}: {}'.format(self.exp_st_index, r.state))
            '''
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
            '''

    fig = plt.figure()
    keys = np.loadtxt(pathfn)
    if swap:
        '''
        Transpose W-Last data into W-First format
        '''
        keys[:, [3,4,5,6]] = keys[:,[6,3,4,5]]
    if interpthresh > 0:
        ikeys = [keys[0]]
        for key in keys[1:]:
            print("Partition {}\n{}".format(ikeys[-1], key))
            inodes = partition(ikeys[-1], key, interpthresh)
            ikeys += inodes
            ikeys.append(key)
        ikeys = np.array(ikeys)
        print("Using: {}".format(ikeys))
        print("Total nodes: {}".format(len(ikeys)))
    ra = ExpandingReAnimator(r, gt, keys)
    ani = animation.FuncAnimation(fig, ra.perform)
    plt.show()

def usage():
    print("rl-reanimator.py <preprocessed ground truth file>")

if __name__ == '__main__':
    np.set_printoptions(linewidth=240)
    # reanimate('blend-low.gt.npz', 'blend.path', swap=False)
    # reanimate('blend-low.gt.npz', 'rrt.path', swap=True, cachedir='blend-traj-16k')
    reanimate('blend-low.gt.npz', 'rrt-secondhalf.path', swap=True, cachedir='blend-traj-secondhalf-parted', interpthresh=4.0)
    # reanimate('blend-low.gt.npz', 'ver1.2.path', swap=True, cachedir='ver1.2-traj')
    #reanimate('blend-low.gt.npz', '../res/alpha/alpha-1.2.org.path', swap=True, cachedir='classical-traj')
    #reanimate('blend-low.gt.npz', '../res/alpha/alpha-1.2.org.path', swap=True, cachedir='classical-traj-parted', interpthresh=1.0)
    # reanimate('blend-low.gt.npz', '../res/alpha/alpha-1.2.org.path', swap=True, cachedir='classical-traj-parted-nonunitmeasure', interpthresh=1.0,in_unit=False)
