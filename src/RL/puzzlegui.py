# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import rlutil
import rlargs
import readline
import pyosr
import uw_random
import numpy as np
from six.moves import input
from rlreanimator import reanimate
import signal
import sys
import os

class PuzzlePlayer(object):
    def __init__(self, args):
        r = rlutil.create_renderer(args)
        # r.set_perturbation(uw_random.random_state(0.25))
        r.state = np.array(r.translate_to_unit_state(args.istateraw), dtype=np.float32)
        self.renderer = r
        self.rgb_shape = (len(r.views), r.pbufferWidth, r.pbufferHeight, 3)
        self.dactions = []
        self.action_magnitude = args.amag
        self.verify_magnitude = args.vmag

    def __iter__(self):
        r = self.renderer
        while True:
            r.render_mvrgbd()
            yield np.copy(r.mvrgb.reshape(self.rgb_shape)[0])
            if not self.dactions:
                texts = input("Action [Number of actions]: ").split()
                if texts:
                    a = int(texts[0])
                    if a < 0:
                        return
                    n = 1
                    if len(texts) > 1:
                        n = int(texts[1])
                    self.dactions += [a] * n
            else:
                action = self.dactions.pop(0)
                nstate, _, ratio = r.transit_state(r.state,
                        action,
                        self.action_magnitude,
                        self.verify_magnitude)
                if ratio < 1e-4:
                    print("Blocked, clear action queue")
                    self.dactions[:] = []
                print("Ratio {}".format(ratio))
                r.state = nstate

class ExplorePlayer(object):
    def __init__(self, args):
        r = rlutil.create_renderer(args)
        self.renderer = r
        self.rgb_shape = (len(r.views), r.pbufferWidth, r.pbufferHeight, 3)
        self.agent_id = -1
        self.frame = 0
        self.stop = True
        signal.signal(signal.SIGINT, self.catch)
        self.frame_generator = None
        aid = 0
        self.aa_tuples = []
        self.view = 0
        while True:
            dn = '{}/agent-{}'.format(args.exploredir, aid)
            if not os.path.isdir(dn):
                break
            fid = 0
            q_list = []
            a_list = []
            tau_list = []
            r_list = []
            term_list = []
            perm_list = []
            while True:
                fn = '{}/{}.npz'.format(dn, fid)
                if not os.path.exists(fn):
                    break
                dic = np.load(fn)
                q_list.append(dic['QSTATE'])
                a_list.append(dic['A'])
                tau_list.append(dic['TAU'])
                r_list.append(dic['R'])
                term_list.append(dic['T'])
                perm_list.append(dic['PERM'])
                fid += 1
            q = np.concatenate(q_list)
            a = np.concatenate(a_list)
            tau = np.concatenate(tau_list)
            r = np.concatenate(r_list)
            term = np.concatenate(term_list)
            perm = np.concatenate(perm_list)
            self.aa_tuples.append((q, a, tau, r, term, perm))
            aid += 1
        print('QSTATE SHAPE {}'.format(self.aa_tuples[self.agent_id][0].shape))

    def __iter__(self):
        r = self.renderer
        while True:
            if self.agent_id < 0:
                self.stop = True
            while self.stop:
                print("Menu: choose an action to perform")
                print("s <num>: select an agent")
                print("f <num>: seek to frame")
                print("v <num>: change view")
                print("p: play")
                print("rp: reverse play")
                print("Status: agent {} frame {} view {}".format(self.agent_id, self.frame, self.view))
                texts = input("> ").split()
                cmd = texts[0]
                if cmd == 's':
                    if len(texts) < 2:
                        continue
                    self.agent_id = int(texts[1])
                elif cmd == 'f':
                    if len(texts) < 2:
                        continue
                    self.frame = int(texts[1])
                elif cmd == 'v':
                    if len(texts) < 2:
                        continue
                    view = int(texts[1])
                    if view >= len(r.views) or view < 0:
                        print("View should be in [0, {})".format(len(r.views)))
                        continue
                    self.view = view
                elif cmd == 'p' or cmd == 'rp':
                    if self.agent_id >= len(self.aa_tuples) or self.agent_id < 0:
                        print("Agent ID should be in [0, {})".format(len(self.aa_tuples)))
                        continue
                    total_frame = len(self.aa_tuples[self.agent_id][0])
                    if self.frame >= total_frame or self.frame < 0:
                        print("Frame # should be in [0, {})".format(len(self.aa_tuples[0])))
                        continue
                    if cmd == 'p':
                        self.frame_generator = iter(range(self.frame, total_frame))
                        self.stop = False
                    else:
                        self.frame_generator = iter(range(-1, self.frame, -1))
                        self.stop = False
            frame = next(self.frame_generator, -1)
            if frame == -1:
                self.stop = True
                continue
            a = self.aa_tuples[self.agent_id][1][frame]
            tau = self.aa_tuples[self.agent_id][2][frame]
            reward = self.aa_tuples[self.agent_id][3][frame]
            print("current frame {} action {} tau {} reward {}".format(frame, a, tau, reward))
            r.state = self.aa_tuples[self.agent_id][0][frame]
            # r.set_perturbation(self.aa_tuples[self.agent_id][5][frame])
            r.render_mvrgbd()
            self.frame = frame
            yield np.copy(r.mvrgb.reshape(self.rgb_shape)[self.view])

    def catch(self, signum, frame):
        self.stop = True

def create_player(args):
    if args.exploredir is None:
        return PuzzlePlayer(args)
    else:
        return ExplorePlayer(args)

def main():
    pyosr.init()
    args = rlargs.parse()
    print(args)
    player = create_player(args)
    reanimate(player)

if __name__ == '__main__':
    main()
