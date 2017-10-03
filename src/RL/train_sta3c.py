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
import time
from rmsprop_applier import RMSPropApplier
from datetime import datetime

init_state = np.array([0.125, -0.075, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
view_config = [(30.0, 12), (-30.0, 12), (0, 4), (90, 1), (-90, 1)]
ckpt_dir = './ta3c-st/ckpt'

class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, loss):
        self.loss = loss

    def begin(self):
          self._step = -1

    def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(self.loss)  # Asks for loss value.

    def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 10 == 0:
            num_examples_per_step = config.BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
            print (format_str % (datetime.now(), self._step, loss_value,
                examples_per_sec, sec_per_batch))

RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
device = "/cpu:0"

def test_rldriver_main():
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    g = tf.Graph()
    with g.as_default():
        grad_applier = RMSPropApplier(learning_rate=tf.placeholder(tf.float32),
                                      decay=RMSP_ALPHA,
                                      momentum=0.0,
                                      epsilon=RMSP_EPSILON,
                                      clip_norm=GRAD_NORM_CLIP,
                                      device=device)
        masterdriver = rldriver.RLDriver(['../res/alpha/env-1.2.obj', '../res/alpha/robot.obj'],
                init_state,
                view_config,
                config.SV_VISCFG,
                config.MV_VISCFG,
                use_rgb=True)
        driver = rldriver.RLDriver(['../res/alpha/env-1.2.obj', '../res/alpha/robot.obj'],
                    init_state,
                    view_config,
                    config.SV_VISCFG,
                    config.MV_VISCFG,
                    use_rgb=True,
                    master_driver=masterdriver,
                    grads_applier=grad_applier)
        driver.get_sync_from_master_op()
        driver.get_apply_grads_op()
        global_step = tf.contrib.framework.get_or_create_global_step()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            while True:
                driver.train_a3c(sess)
        '''
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=ckpt_dir,
                hooks=[tf.train.StopAtStepHook(last_step=config.MAX_STEPS),
                    tf.train.NanTensorHook(driver.get_total_loss()),
                    _LoggerHook(driver.get_total_loss())]) as mon_sess:
            epoch = 0
            while not mon_sess.should_stop():
                driver.train_a3c(mon_sess)
                epoch += 1
        '''

if __name__ == '__main__':
    test_rldriver_main()
