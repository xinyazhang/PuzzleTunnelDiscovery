import pyosr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imsave
import vision
import tensorflow as tf
import rldriver
import config
import threading

def test_rldriver_worker(dpy, glctx, masterdriver):
    pyosr.create_gl_context(dpy, glctx) # OpenGL context for current thread.
    driver = rldriver.RLDriver(['../res/alpha/env-1.2.obj', '../res/alpha/robot.obj'],
                np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                [(30.0, 12), (-30.0, 12), (0, 4), (90, 1), (-90, 1)],
                config.SV_VISCFG,
                config.MV_VISCFG,
                6,
                use_rgb = True,
                master_driver=masterdriver)
    r = driver.renderer
    w = r.pbufferWidth
    h = r.pbufferHeight
    r.state = np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    r.render_mvrgbd()
    print('worker mvrgb shape {}'.format(r.mvrgb.shape))
    img = r.mvrgb.reshape(w * r.views.shape[0], h, 3)
    dep = r.mvdepth.reshape(w * r.views.shape[0], h)
    depimg = Image.fromarray(dep)
    imsave('mvrgb-worker.png', img)
    depimg.save('mvdepth-worker.tiff')

def test_rldriver_main():
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    init_state = np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    with tf.Graph().as_default():
        driver = rldriver.RLDriver(['../res/alpha/env-1.2.obj', '../res/alpha/robot.obj'],
                np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                [(30.0, 12), (-30.0, 12), (0, 4), (90, 1), (-90, 1)],
                config.SV_VISCFG,
                config.MV_VISCFG,
                6,
                use_rgb=True)
    thread = threading.Thread(target=test_rldriver_worker, args=(dpy, glctx, driver))
    thread.start()
    r = driver.renderer
    w = r.pbufferWidth
    h = r.pbufferHeight
    r.state = np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    r.render_mvrgbd()
    img = r.mvrgb.reshape(w * r.views.shape[0], h, 3)
    dep = r.mvdepth.reshape(w * r.views.shape[0], h)
    depimg = Image.fromarray(dep)
    imsave('mvrgb-master.png', img)
    depimg.save('mvdepth-master.tiff')
    thread.join()

if __name__ == '__main__':
    test_rldriver_main()
