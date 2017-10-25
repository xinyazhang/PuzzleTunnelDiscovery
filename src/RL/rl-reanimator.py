import pyosr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import aniconf12 as aniconf
import sys

def interpolate(pkey, nkey, tau):
    return pyosr.interpolate(pkey, nkey, tau)

def reanimate(gtfn):
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
    gt.rl_stepping_size = 0.0125 / 4
    gt.verify_magnitude = 0.0125 / 8
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

        def __init__(self, renderer, keys, delta=0.125):
            self.renderer = renderer
            self.keys = keys
            self.t = 0.0
            self.delta = delta
            self.reaching_terminal = False

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
                print(r.mvrgb.shape)
                rgb = r.mvrgb.reshape((r.pbufferWidth, r.pbufferHeight, 3))
                print(r.state)
                valid = r.is_valid_state(r.state)
                print('\tNew State {} Collision Free ? {}'.format(r.state, valid))
                if not valid:
                    print('\tNOT COLLISION FREE, SAN CHECK FAILED AT {}'.format(self.t))
                    self.reaching_terminal = True
                if self.im is None:
                    print('rgb {}'.format(rgb.shape))
                    self.im = plt.imshow(rgb)
                else:
                    self.im.set_array(rgb)
                self.t += self.delta

    fig = plt.figure()
    keys = np.loadtxt(aniconf.keys_fn)
    keys[:, [3,4,5,6]] = keys[:,[6,3,4,5]]
    init_state = keys[0]
    gtstats, actions, terminated = gt.generate_gt_path(init_state, 64)
    np.savez('data-from-rl-reanimator.tmp', S=gtstats, A=actions)
    # print('trajectory:\n')
    # print(gtstats)
    print("terminated ? {} Expecting True".format(terminated))
    ra = ReAnimator(r, gtstats, 1.0)
    ani = animation.FuncAnimation(fig, ra.perform)
    plt.show()

def usage():
    print("rl-reanimator.py <preprocessed ground truth file>")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        exit()
    reanimate(sys.argv[1])
