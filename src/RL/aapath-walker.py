import pyosr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import aniconf12 as aniconf
import sys
import os.path

kEnableActionCalculation = True
kEpsilon = 1e-6

def interpolate(pkey, nkey, tau):
    return pyosr.interpolate(pkey, nkey, tau)

def reanimate(gtfn, pathfn):
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
    gt.rl_stepping_size = 0.0125 / 16
    gt.verify_magnitude = 0.0125 / 16 / 8
    # gt.rl_stepping_size = 0.0125 / 16 / 1024
    # gt.verify_magnitude = 0.0125 / 16 / 1024 / 2
    # gt.rl_stepping_size = 0.0125 / 64
    # gt.verify_magnitude = 0.0125 / 64 / 8
    gtdata = np.load(gtfn)
    gt.install_gtdata(gtdata['V'], gtdata['E'], gtdata['D'], gtdata['N'])

    # FIXME: add this back after debugging Dijkstra
    # gt.init_knn_in_batch()

    class ReAnimator(object):
        reaching_terminal = False
        driver = None
        im = None
        cont_actions = None

        def __init__(self, renderer, init_state, cont_tr, cont_rot, delta=0.125):
            self.renderer = renderer
            self.cont_tr = cont_tr
            self.cont_rot = cont_rot
            self.t = 0.0
            assert delta <= 1.0, "delta must be <= 1.0, we are reanimating with stateful ACTIONS"
            self.delta = delta
            self.state = renderer.translate_to_unit_state(init_state)
            self.reaching_terminal = False

        def perform(self, framedata):
            r = self.renderer
            index = int(math.floor(self.t))
            if index < len(self.cont_tr) and not self.reaching_terminal:
                pkey = self.state
                nkey,_,_ = r.transit_state_by(pkey, self.cont_tr[index], self.cont_rot[index], gt.verify_magnitude)
                tau = self.t - index
                print("tau {}".format(tau))
                state = interpolate(pkey, nkey, tau)
                r.state = state
                r.render_mvrgbd()
                print(r.mvrgb.shape)
                rgb = r.mvrgb.reshape((r.pbufferWidth, r.pbufferHeight, 3))
                print(r.state)
                valid = r.is_valid_state(r.state)
                disentangled = r.is_disentangled(r.state)
                print('\tNew State {} Collision Free ? {} Disentangled ? {}'.format(r.state, valid, disentangled))
                if not valid:
                    print('\tNOT COLLISION FREE, SAN CHECK FAILED')
                    self.reaching_terminal = True
                if self.im is None:
                    print('rgb {}'.format(rgb.shape))
                    self.im = plt.imshow(rgb)
                else:
                    self.im.set_array(rgb)
                self.t += self.delta
                if index != int(math.floor(self.t)):
                    self.state = nkey


    fig = plt.figure()
    keys = np.loadtxt(pathfn)
    fn = 'blend-cont-saved.npz'
    if not os.path.isfile(fn):
        print("Translating into Cont. Actions...")
        cont_tr, cont_rot, _ = gt.cast_path_to_cont_actions_in_UW(keys)
        print("Done")
        np.savez('blend-cont.npz', TR=cont_tr, ROT=cont_rot)
    else:
        print("Loading cacahed Cont. Actions")
        dic = np.load(fn)
        cont_tr = dic['TR']
        cont_rot = dic['ROT']
    # keys[:, [3,4,5,6]] = keys[:,[6,3,4,5]]
    ra = ReAnimator(r, keys[0], cont_tr, cont_rot, 1.0)
    ani = animation.FuncAnimation(fig, ra.perform)
    plt.show()

def usage():
    print("aapath-walker.py <preprocessed ground truth file>")

if __name__ == '__main__':
    reanimate('blend-low.gt.npz', 'blend.path')
