import pyosr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imsave
import vision
import tensorflow as tf
import rldriver
import config

def test():
    pyosr.init()
    pyosr.create_gl_context(pyosr.create_display())

    r=pyosr.Renderer()
    r.setup()
    r.loadModelFromFile('../res/alpha/env-1.2.obj')
    r.loadRobotFromFile('../res/alpha/robot.obj')
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    r.views = np.array([[30.0, float(i)] for i in range(0, 360, 30)], dtype=np.float32)
    print(r.views)
    print(r.views.shape)
    w = r.pbufferWidth
    h = r.pbufferHeight
    r.state = np.array([200.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    r.render_mvrgbd()
    img = r.mvrgb.reshape(w * r.views.shape[0], h, 3)
    dep = r.mvdepth.reshape(w * r.views.shape[0], h)
    depimg = Image.fromarray(dep)
    imsave('mvrgb.png', img)
    depimg.save('mvdepth.tiff')
    print("Is colliding free? {} {}".format(r.state, r.is_valid_state(r.state)))
    r.state = np.array([0.04, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    print("Is colliding free? {} {} Expecting False".format(r.state, r.is_valid_state(r.state)))
    r.state = np.array([0.7, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    print("Is colliding free? {} {} Expecting True".format(r.state, r.is_valid_state(r.state)))
    exit()
    for x in np.arange(-0.8, 0.8, 0.02):
        r.state = np.array([x, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        print("Is colliding free? {} {}".format(r.state, r.is_valid_state(r.state)))
        r.render_mvrgbd()
        # print('r.mvrgb {}'.format(r.mvrgb.shape))
        # print('r.views {} w {} h {}'.format(r.views.shape, w, h))
        img = r.mvrgb.reshape(w * r.views.shape[0], h, 3)
        imsave('mvrgb-x={}.png'.format(x), img)

    r.teardown()
    pyosr.shutdown()

if __name__ == '__main__':
    test()
