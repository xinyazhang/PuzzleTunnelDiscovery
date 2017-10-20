import pyosr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import aniconf12 as aniconf

def interpolate(pkey, nkey, tau):
    return pyosr.interpolate(pkey, nkey, tau)

def reanimate():
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
                    print('\tNOT COLLISION FREE, SAN CHECK FAILED')
                    self.reaching_terminal = True
                if self.im is None:
                    print('rgb {}'.format(rgb.shape))
                    self.im = plt.imshow(rgb)
                else:
                    self.im.set_array(rgb)
                self.t += self.delta

    fig = plt.figure()
    keys = np.loadtxt(aniconf.keys_fn)
    print('before keys[0] {}'.format(keys[0]))
    keys[:, [3,4,5,6]] = keys[:,[6,3,4,5]]
    print('after keys[0] {}'.format(keys[0]))
    ra = ReAnimator(r, keys, 1.0)
    ani = animation.FuncAnimation(fig, ra.perform)
    plt.show()

if __name__ == '__main__':
    reanimate()
