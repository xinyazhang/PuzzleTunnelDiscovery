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
import util
from rmsprop_applier import RMSPropApplier
from datetime import datetime

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
device = "/gpu:0"
MODELS = ['../res/simple/FullTorus.obj', '../res/simple/robot.obj']
init_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
view_config = [(30.0, 12), (-30.0, 12), (0, 4), (90, 1), (-90, 1)]
ckpt_dir = './ttorus/ckpt/'
ckpt_prefix = 'torus-vs-ring-ckpt'

def test_rldriver_main():
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    g = tf.Graph()
    util.mkdir_p(ckpt_dir)
    with g.as_default():
        learning_rate_input = tf.placeholder(tf.float32)
        grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                                      decay=RMSP_ALPHA,
                                      momentum=0.0,
                                      epsilon=RMSP_EPSILON,
                                      clip_norm=GRAD_NORM_CLIP,
                                      device=device)
        masterdriver = rldriver.RLDriver(MODELS,
                init_state,
                view_config,
                config.SV_VISCFG,
                config.MV_VISCFG,
                use_rgb=True)
        driver = rldriver.RLDriver(MODELS,
                    init_state,
                    view_config,
                    config.SV_VISCFG,
                    config.MV_VISCFG,
                    use_rgb=True,
                    master_driver=masterdriver,
                    grads_applier=grad_applier)
        driver.get_sync_from_master_op()
        driver.get_apply_grads_op()
        driver.learning_rate_input = learning_rate_input
        driver.a3c_local_t = 32
        global_step = tf.contrib.framework.get_or_create_global_step()
        increment_global_step = tf.assign_add(global_step, 1, name='increment_global_step')
        saver = tf.train.Saver(masterdriver.get_nn_args() + [global_step])
        last_time = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
            print('ckpt {}'.format(ckpt))
            epoch = 0
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                epoch = sess.run(global_step)
                print('Restored!, global_step {}'.format(epoch))
            while epoch < 100 * 1000:
                driver.train_a3c(sess)
                epoch += 1
                sess.run(increment_global_step)
                if epoch % 1000 == 0 or time.time() - last_time >= 10 * 60:
                    print("Saving checkpoint")
                    fn = saver.save(sess, ckpt_dir+ckpt_prefix, global_step=global_step)
                    print("Saved checkpoint to {}".format(fn))
                    last_time = time.time()
                print("Epoch {}".format(epoch))
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
