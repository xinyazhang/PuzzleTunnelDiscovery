# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
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
import random
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
ckpt_dir = './ttorus/ckpt-mt-6/'
ckpt_prefix = 'torus-vs-ring-ckpt'
THREAD = 1
POLICIES = [0.2, 0.4, 0.5, 0.6, 0.8]

graph_completes = [threading.Event() for i in range(THREAD)]
init_done = threading.Event()

def torus_worker(index, dpy, glctx, masterdriver, tfgraph, grad_applier, lrtensor, global_step, increment_global_step, sess, saver):
    global graph_completes
    global init_done
    pyosr.create_gl_context(dpy, glctx) # OpenGL context for current thread.
    with tfgraph.as_default():
        driver = rldriver.RLDriver(MODELS,
                    init_state,
                    view_config,
                    config.SV_VISCFG,
                    config.MV_VISCFG,
                    use_rgb=True,
                    master_driver=masterdriver,
                    grads_applier=grad_applier,
                    worker_thread_index=index)
        print("THREAD {} DRIVER CREATED".format(index))
        driver.epsilon = 1.0 - (index + 1) * (1.0 / (THREAD + 1))
        driver.get_sync_from_master_op()
        driver.get_apply_grads_op()
        driver.learning_rate_input = lrtensor
        # driver.a3c_local_t = 32
        graph_completes[index].set()
        init_done.wait()
        '''
        if index == 0:
            for i in range(1,4):
                graph_completes[i].wait()
                print("Graph {} waited".format(i))
            sess.run(tf.global_variables_initializer())
            init_done.set()
        else:
            graph_completes[index].set()
            print("Graph {} Set".format(index))
            init_done.wait()
            print("Init_done on thread {}, continuing".format(index))
        '''
        last_time = time.time()
        # FIXME: ttorus/ckpt-mt-5 are not loading global_step into epoch
        epoch = 0
        driver.verbose_training = True
        while epoch < 100 * 1000:
        #while epoch < 2 * 1000:
            driver.epsilon = random.choice(POLICIES)
            driver.train_a3c(sess)
            epoch += 1
            sess.run(increment_global_step)
            if index == 0 and time.time() - last_time >= 60 * 10:
                print("Saving checkpoint")
                fn = saver.save(sess, ckpt_dir+ckpt_prefix, global_step=global_step)
                print("Saved checkpoint to {}".format(fn))
                last_time = time.time()
            print("[{}] Epoch {}, global_step {}".format(index, epoch, sess.run(global_step)))
            if index == 0:
                sess.run(driver.get_sync_from_master_op())
                driver.renderer.state = init_state
                _, value, _, _ = driver.evaluate(sess)
                print("Master Driver V for init_state {}".format(value))
                # Choose some random initial conf for better training
                if random.random() > driver.epsilon:
                    driver.restart_epoch()
            else:
                driver.restart_epoch()

def torus_master():
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    g = tf.Graph()
    util.mkdir_p(ckpt_dir)
    with g.as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        increment_global_step = tf.assign_add(global_step, 1, name='increment_global_step')
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
        saver = tf.train.Saver(masterdriver.get_nn_args() + [global_step])
        with tf.Session() as sess:
            threads = []
            for i in range(THREAD):
                thread_args = (i, dpy, glctx, masterdriver, g, grad_applier,
                        learning_rate_input, global_step,
                        increment_global_step, sess, saver)
                thread = threading.Thread(target=torus_worker, args=thread_args)
                thread.start()
                graph_completes[i].wait()
                threads.append(thread)
            '''
            We need to run the initializer because only master's variables are stored.
            '''
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
            print('ckpt {}'.format(ckpt))
            epoch = 0
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                epoch = sess.run(global_step)
                print('Restored!, global_step {}'.format(epoch))

            init_done.set()
            for thread in threads:
                thread.join()
            print("Saving final checkpoint")
            fn = saver.save(sess, ckpt_dir+ckpt_prefix, global_step=global_step)
            print("Saved checkpoint to {}".format(fn))

if __name__ == '__main__':
    torus_master()
