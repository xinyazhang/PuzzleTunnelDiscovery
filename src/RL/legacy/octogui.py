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

class OctoPlayer(object):
    def __init__(self, args):
        r = rlutil.create_renderer(args)
        # r.set_perturbation(uw_random.random_state(0.25))
        r.state = np.array(r.translate_to_unit_state(args.istateraw), dtype=np.float32)
        self.renderer = r
        self.rgb_shape = (len(r.views), r.pbufferWidth, r.pbufferHeight, 3)
        self.dactions = []
        self.action_magnitude = args.amag
        self.verify_magnitude = args.vmag
        self.target = None

    def __iter__(self):
        r = self.renderer
        while True:
            r.render_mvrgbd()
            yield np.copy(r.mvrgb.reshape(self.rgb_shape)[0])
            if self.target is None:
                texts = input("Goal State (W-Last)").split()
                self.target = np.array(texts, dtype=np.float64)
                self.target[[3,4,5,6]] = self.target[[6,3,4,5]]
                self.target = r.translate_to_unit_state(self.target)
            DA = uw_random.DISCRETE_ACTION_NUMBER
            ns = np.zeros((DA,7))
            d  = np.zeros((DA))
            for action in range(DA):
                nstate, _, ratio = r.transit_state(r.state,
                        action,
                        self.action_magnitude,
                        self.verify_magnitude)
                ns[action] = nstate
                if ratio < 1e-4:
                    d[action] = 999.9
                else:
                    d[action] = pyosr.distance(nstate, self.target)
                print("\tA {} NS {} Ratio {} D {}".format(action, nstate, ratio, d[action]))
            best = np.argmax(d)
            bstate = ns[best]
            r.state = bstate

def main():
    pyosr.init()
    args = rlargs.parse()
    print(args)
    player = OctoPlayer(args)
    reanimate(player)

if __name__ == '__main__':
    main()
