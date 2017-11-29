import pyosr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import aniconf12 as aniconf
import uw_random

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
    r.views = np.array([[15.0, 110.0]], dtype=np.float32)
    print(r.scene_matrix)
    print(r.robot_matrix)
    r.set_perturbation(uw_random.random_state(0.00))
    # r.set_perturbation(np.array([0,0.0,0,0.5,0.5,0.5,0.5],dtype=np.float32))
    # r.set_perturbation(np.array([0,0.0,0,0,0,1,0],dtype=np.float32))
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

    fig = plt.figure()
    keys = np.loadtxt(aniconf.keys_fn)
    if aniconf.keys_w_last:
        print('before keys[0] {}'.format(keys[0]))
        keys[:, [3,4,5,6]] = keys[:,[6,3,4,5]]
        print('after keys[0] {}'.format(keys[0]))
    ra = ReAnimator(r, keys, 1.0)
    '''
    st0 = np.array([21.782108575648873,11.070742691783639,13.072090341969885,0.99496368307688909,-0.050573680994590003,0.08255004745739393,0.025981951687884433], dtype=np.float64)
    st1 = np.array([24.404447383193428,16.614281021136808,17.241012748680941,0.89856334412742611,-0.42392368380753659,0.035352511370216902,0.10780921499298371], dtype=np.float64)
    st2 = np.array([25.04292893291256,16.429006629785405,21.742598419634952,-0.080755517119222811,-0.980264716314169,0.167639957003235,0.066756851485538532], dtype=np.float64)
    ra = ReAnimator(r, [st1, st2], 0.0125)
    '''
    '''
    st1 = np.array([21.75988005530629,19.55840991214458,18.407298399116954,0.8747148780179097,-0.40598294544114955,0.21522777832862061,0.15404133737658857], dtype=np.float64)
    st2 = np.array([16.242401343877191,15.371074546390151,23.775856491398514,0.38753152804205182,-0.26626876971877833,-0.75270143169934201,-0.46082622729571437], dtype=np.float64)
    ra = ReAnimator(r, [st1, st2], 0.0125/2.0)
    '''
    ani = animation.FuncAnimation(fig, ra.perform)
    plt.show()

if __name__ == '__main__':
    reanimate()
