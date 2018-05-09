import rlutil
import rlargs
import readline
import pyosr
import uw_random
import numpy as np
from six.moves import input
from rlreanimator import reanimate

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

def main():
    pyosr.init()
    args = rlargs.parse()
    print(args)
    player = PuzzlePlayer(args)
    reanimate(player)

if __name__ == '__main__':
    main()
