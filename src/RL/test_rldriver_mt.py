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

def test_rldriver_worker(dpy, glctx, masterdriver, tfgraph):
    pyosr.create_gl_context(dpy, glctx) # OpenGL context for current thread.
    with tfgraph.as_default():
        driver = rldriver.RLDriver(['../res/alpha/env-1.2.obj', '../res/alpha/robot.obj'],
                    np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                    [(30.0, 12), (-30.0, 12), (0, 4), (90, 1), (-90, 1)],
                    config.SV_VISCFG,
                    config.MV_VISCFG,
                    use_rgb = True,
                    master_driver=masterdriver)
        sync_op = driver.get_sync_from_master_op()
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
    with tfgraph.as_default():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            value_before = np.array(sess.run(driver.get_nn_args()[0][0][0]), dtype=np.float32)
            print('Before {}'.format(value_before))
            sess.run(sync_op)
            value_after = np.array(sess.run(driver.get_nn_args()[0][0][0]), dtype=np.float32)
            print('After {}'.format(value_after))
            print('Delta {}'.format(np.linalg.norm(value_before - value_after)))

'''
Torus final state
0.11249993  0.02 0.12499992  0.99983597  0.01287497  0.00606527 0.01132157
'''

def test_rldriver_main():
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    init_state = np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    g = tf.Graph()
    with g.as_default():
        driver = rldriver.RLDriver(['../res/alpha/env-1.2.obj', '../res/alpha/robot.obj'],
                np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                [(30.0, 12), (-30.0, 12), (0, 4), (90, 1), (-90, 1)],
                config.SV_VISCFG,
                config.MV_VISCFG,
                use_rgb=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    thread = threading.Thread(target=test_rldriver_worker, args=(dpy, glctx, driver, g))
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
