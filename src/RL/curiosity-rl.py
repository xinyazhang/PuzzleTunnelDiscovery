'''
    pretrain.py

    Pre-Train the VisionNet and Inverse Model
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import aniconf12 as aniconf
import matplotlib.pyplot as plt
from scipy.misc import imsave
import matplotlib.animation as animation
import sys
import os
import time
import util
import argparse
import uw_random
import config
import vision
import pyosr
import icm
import rlenv
import Queue as queue # Python 2, rename to import queue as queue for python 3
import rlargs
import a2c
import random
import threading
from cachetools import LRUCache
from six.moves import queue

MT_VERBOSE = False
# MT_VERBOSE = True
VIEW_CFG = config.VIEW_CFG

COLLIDE_PEN_MAG = 1e3

def setup_global_variable(args):
    global VIEW_CFG
    if args.viewset == 'cube':
        VIEW_CFG = [(0, 4), (90, 1), (-90, 1)]
    elif args.viewset == '14' or args.ferev >= 4:
        VIEW_CFG = config.VIEW_CFG_REV4
    elif args.viewset == '22' or args.ferev != 1:
        VIEW_CFG = config.VIEW_CFG_REV2

def _get_action_set(args):
    if args.uniqueaction > 0:
        return [args.uniqueaction]
    return args.actionset

def _get_view_cfg(args):
    view_array = vision.create_view_array_from_config(VIEW_CFG)
    if args.view >= 0:
        view_num = 1
    else:
        view_num = len(view_array)
    return view_num, view_array

def create_renderer(args):
    w = h = args.res
    view_num, view_array = _get_view_cfg(args)

    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    r = pyosr.Renderer()
    r.pbufferWidth = w
    r.pbufferHeight = h
    r.setup()
    r.loadModelFromFile(aniconf.env_fn)
    r.loadRobotFromFile(aniconf.rob_fn)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    if args.view >= 0:
        r.views = np.array([view_array[args.view]], dtype=np.float32)
    else:
        r.views = np.array(view_array, dtype=np.float32)
    return r

class GroundTruth:
    pass

class AlphaPuzzle(rlenv.IEnvironment):
    collision_pen_mag = COLLIDE_PEN_MAG
    solved_award_mag = 1e7

    def __init__(self, args, tid):
        super(AlphaPuzzle, self).__init__()
        self.fb_cache = None
        self.fb_dirty = True
        r = self.r = create_renderer(args)
        self.istate = np.array(r.translate_to_unit_state(args.istateraw), dtype=np.float32)
        r.state = self.istate
        self.rgb_shape = (len(r.views), r.pbufferWidth, r.pbufferHeight, 3)
        self.dep_shape = (len(r.views), r.pbufferWidth, r.pbufferHeight, 1)
        self.action_magnitude = args.amag
        self.verify_magnitude = args.vmag
        self.collision_cache = LRUCache(maxsize = 128)
        self.egreedy = args.egreedy[0] # e-greedy is an agent-specific variable
        if len(args.egreedy) != 1:
            self.egreedy = args.egreedy[tid]

    def qstate_setter(self, state):
        # print('old {}'.format(self.r.state))
        # print('new {}'.format(state))
        if np.array_equal(self.r.state, state):
            return
        self.r.state = state
        self.fb_dirty = True

    def qstate_getter(self):
        return self.r.state

    qstate = property(qstate_getter, qstate_setter)

    @property
    def vstate(self):
        if self.fb_cache is not None and not self.fb_dirty:
            return self.fb_cache
        self.fb_dirty = False
        r = self.r
        r.render_mvrgbd()
        rgb = np.copy(r.mvrgb.reshape(self.rgb_shape))
        dep = np.copy(r.mvdepth.reshape(self.dep_shape))
        self.fb_cache = [rgb, dep]
        return self.fb_cache

    @property
    def vstatedim(self):
        return self.rgb_shape[0:3]

    def peek_act(self, action, pprefix=""):
        r = self.r
        colkey = tuple(r.state.tolist() + [action])
        if colkey in self.collision_cache:
            print("Cache Hit {}".format(colkey))
            nstate, done, ratio = self.collision_cache[colkey]
        else:
            nstate, done, ratio = r.transit_state(r.state,
                    action,
                    self.action_magnitude,
                    self.verify_magnitude)
            if ratio < 1e-4:
                '''
                Disable moving forward if ratio is too small.
                '''
                ratio = 0
                nstate = np.copy(r.state)
        sa = (colkey, (nstate, done, ratio))
        reaching_terminal = r.is_disentangled(nstate)
        reward = 0.0
        reward += self.solved_award_mag if reaching_terminal is True else 0.0 # Large Reward for solution
        if not done:
            '''
            Special handling of collision
            '''
            if ratio == 0.0:
                reward = -self.collision_pen_mag
            self.collision_cache.update([sa])
        rgb_1, dep_1 = self.vstate
        self.state = nstate
        rgb_2, dep_2 = self.vstate
        print(pprefix, "New state {} ratio {} terminal {} reward {}".format(nstate, ratio, reaching_terminal, reward))
        return nstate, reward, reaching_terminal

    def reset(self):
        self.qstate = self.istate

class CuriosityRL(rlenv.IAdvantageCore):
    rgb_shape = None
    dep_shape = None
    fb_dirty = True
    fb_cache = None
    polout = None
    polparams = None
    action_magnitude = None
    verify_magnitude = None

    def __init__(self, learning_rate, args):
        super(CuriosityRL, self).__init__()
        self.view_num, _ = _get_view_cfg(args)
        w = h = args.res

        self.action_tensor = tf.placeholder(tf.float32, shape=[None, 1, uw_random.DISCRETE_ACTION_NUMBER], name='ActionPh')
        self.rgb_1_tensor = tf.placeholder(tf.float32, shape=[None, self.view_num, w, h, 3], name='Rgb1Ph')
        self.rgb_2_tensor = tf.placeholder(tf.float32, shape=[None, self.view_num, w, h, 3], name='Rgb2Ph')
        self.dep_1_tensor = tf.placeholder(tf.float32, shape=[None, self.view_num, w, h, 1], name='Dep1Ph')
        self.dep_2_tensor = tf.placeholder(tf.float32, shape=[None, self.view_num, w, h, 1], name='Dep2Ph')

        if self.view_num > 1:
            self.model = icm.IntrinsicCuriosityModuleIndependentCommittee(self.action_tensor,
                    self.rgb_1_tensor, self.dep_1_tensor,
                    self.rgb_2_tensor, self.dep_2_tensor,
                    config.SV_VISCFG,
                    config.MV_VISCFG2,
                    args.featnum,
                    args.elu,
                    args.ferev,
                    args.imhidden,
                    args.fehidden)
        else:
            with tf.variable_scope(icm.view_scope_name(args.view)):
                self.model = icm.IntrinsicCuriosityModule(self.action_tensor,
                        self.rgb_1, self.dep_1,
                        self.rgb_2, self.dep_2,
                        config.SV_VISCFG,
                        config.MV_VISCFG2,
                        featnum=args.featnum,
                        elu=args.elu,
                        ferev=args.ferev,
                        imhidden=args.imhidden,
                        fehidden=args.fehidden,
                        fwhidden=args.fwhidden)
                self.model.get_inverse_model()

        self.inverse_loss = self.model.get_inverse_loss(discrete=True)

        self.polout, self.polparams, self.polnets = self.create_polnet(args)
        self.valout, self.valparams, self.valnets = self.create_valnet(args)
        '''
        valout ranges from -2 COLLIDE_PEN_MAG to + inf
        '''
        self.valout = self.valout * (COLLIDE_PEN_MAG * 2.0)
        self.curiosity, self.curiosity_params = self.create_curiosity_net(args)
        print('Curiosity Params: {}'.format(self.curiosity_params))

        self.using_lstm = args.lstm
        if args.lstm:
            self.lstm_states_in, self.lstm_len, self.lstm_states_out = self.model.acquire_lstm_io('LSTM')
            # batching is madantory, although our batch size is 1
            self.current_lstm = tf.contrib.rnn.LSTMStateTuple(
                    np.zeros([1, self.model.lstmsize], dtype=np.float32),
                    np.zeros([1, self.model.lstmsize], dtype=np.float32))
            print("[LSTM] {}".format(self.lstm_states_in.c))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.refine_vision_and_memory_op = self.optimizer.minimize(self.inverse_loss + self.curiosity,
                var_list = self.curiosity_params)

    def load_pretrained(self, args):
        self.icm.restore(sess, args.viewinitckpt)

    def create_polnet(self, args):
        hidden = args.polhidden + [int(self.action_tensor.shape[-1])]
        return self.model.create_somenet_from_feature(hidden, 'PolNet',
                elu=args.elu,
                lstm=args.lstm,
                initialized_as_zero=True)

    def create_valnet(self, args):
        hidden = args.valhidden + [1]
        return self.model.create_somenet_from_feature(hidden, 'ValNet',
                elu=args.elu,
                lstm=args.lstm,
                initialized_as_zero=True)

    def create_curiosity_net(self, args):
        fwd_params, fwd_feat = self.model.get_forward_model()
        '''
        Note: we need to train the curiosity model as memory, so tf.losses is used.
        '''
        curiosity = tf.losses.mean_squared_error(fwd_feat, self.model.next_featvec)
        return curiosity, fwd_params

    @property
    def rgb_1(self):
        return self.rgb_1_tensor
    @property
    def rgb_2(self):
        return self.rgb_2_tensor
    @property
    def dep_1(self):
        return self.dep_1_tensor
    @property
    def dep_2(self):
        return self.dep_2_tensor
    @property
    def policy(self):
        return self.polout
    @property
    def value(self):
        return self.valout
    @property
    def policy_params(self):
        return self.polparams
    @property
    def value_params(self):
        return self.valparams
    @property
    def lstm_params(self):
        if not self.using_lstm:
            return []
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LSTM')

    def evaluate_current(self, envir, sess, tensors, additional_dict=None):
        rgb,dep = envir.vstate
        # NOTE: all nets accepts multiple frames, so we use lists of 1 element for single frame
        dic = {
                self.rgb_1 : [rgb],
                self.dep_1 : [dep]
              }
        if additional_dict is not None:
            dic.update(additional_dict)
        if not self.using_lstm:
            return sess.run(tensors, feed_dict=dic)
        # print(self.lstm_states_in.c)
        # print(self.current_lstm)
        dic.update({
            self.lstm_states_in.c : self.current_lstm.c,
            self.lstm_states_in.h : self.current_lstm.h,
            self.lstm_len : 1
                   })
        # FIXME: verify ret[-1] is a tuple
        ret = sess.run(tensors + [self.lstm_states_out], feed_dict=dic)
        self.lstm_cache = ret[-1]
        return ret[:-1]

    def make_decision(self, envir, policy_dist, pprefix=''):
        best = np.argmax(policy_dist, axis=-1)
        if random.random() < envir.egreedy:
            ret = random.randrange(uw_random.DISCRETE_ACTION_NUMBER)
        else:
            ret = best
        print(pprefix, 'Action best {} chosen {}'.format(best, ret))
        return ret

    def get_artificial_reward(self, envir, sess, state_1, adist, state_2, pprefix=""):
        envir.qstate = state_1
        vs1 = envir.vstate
        envir.qstate = state_2
        vs2 = envir.vstate
        dic = {
                self.action_tensor : [[adist]], # Expand from [12] (A only) to [1,1,12] ([F,V,A])
                self.rgb_1 : [vs1[0]],
                self.dep_1 : [vs1[1]],
                self.rgb_2 : [vs2[0]],
                self.dep_2 : [vs2[1]],
              }
        ret = sess.run(self.curiosity, feed_dict=dic)
        print(pprefix, "AR {}".format(ret))
        print(pprefix, "AR Input adist {} vs1 {} {} vs2 {} {}".format(adist, vs1[0].shape, vs1[1].shape, vs2[0].shape, vs2[1].shape))
        return ret

    def train(self, sess, rgb, dep, actions):
        dic = {
                self.action_tensor : actions,
                self.rgb_1 : rgb[:-1],
                self.dep_1 : dep[:-1],
                self.rgb_2 : rgb[1:],
                self.dep_2 : dep[1:],
              }
        sess.run(self.refine_vision_and_memory_op, feed_dict=dic)

    def lstm_next(self):
        if self.using_lstm:
            return self.lstm_cache
        return 0

    def set_lstm(self, lstm):
        if self.using_lstm:
            self.current_lstm = lstm

    def get_lstm(self):
        if self.using_lstm:
            return self.current_lstm
        return 0

    def load_pretrain(self, sess, viewinitckpt):
        if self.view_num > 1:
            self.model.load_pretrain(sess, viewinitckpt)
        else:
            self.model.load_pretrain(sess, viewinitckpt[0])

class TrainerMT:
    kAsyncTask = 1
    kSyncTask = 2
    kExitTask = -1

    def __init__(self, args, g, global_step):
        if len(args.egreedy) != 1 and len(args.egreedy) != args.threads:
            print("--egreedy should have only one argument, or match the number of threads")
        self.args = args
        self.advcore = CuriosityRL(learning_rate=1e-3, args=args)
        self.tfgraph = g
        self.threads = []
        self.taskQ = queue.Queue(args.queuemax)
        self.sessQ = queue.Queue(args.queuemax)
        self.reportQ = queue.Queue(args.queuemax)
        self.trainer = a2c.A2CTrainer(
                advcore=self.advcore,
                tmax=args.batch,
                gamma=config.GAMMA,
                learning_rate=1e-3,
                global_step=global_step)
        for i in range(args.threads):
            thread = threading.Thread(target=self.run_worker, args=(i,g))
            thread.start()
            self.threads.append(thread)

    def run_worker(self, tid, g):
        '''
        IEnvironment is pre-thread object, mainly due to OpenGL context
        '''
        with g.as_default():
            args = self.args
            envir = AlphaPuzzle(args, tid)
            while True:
                task = self.taskQ.get()
                if task == self.kExitTask:
                    return 0
                sess = self.sessQ.get()
                self.trainer.train(envir, sess, tid)
                if task == self.kSyncTask:
                    self.reportQ.put(1)

    def train(self, sess, is_async=True):
        self.sessQ.put(sess)
        if is_async:
            self.taskQ.put(self.kAsyncTask)
        else:
            self.taskQ.put(self.kSyncTask)
            done = self.reportQ.get()

    def stop(self):
        for t in self.threads:
            self.taskQ.put(self.kExitTask)
        for t in self.threads:
            t.join()


def curiosity_main(args):
    '''
    CAVEAT: WITHOUT ALLOW_GRWTH, WE MUST CREATE RENDERER BEFORE CALLING ANY TF ROUTINE
    '''
    pyosr.init()
    total_epoch = args.total_epoch

    view_num, view_array = _get_view_cfg(args)
    w = h = args.res

    ckpt_dir = args.ckptdir
    ckpt_prefix = args.ckptprefix
    device = args.device

    if 'gpu' in device:
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # session_config = tf.ConfigProto(gpu_options=gpu_options)
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
    else:
        session_config = None

    g = tf.Graph()
    with g.as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        increment_global_step = tf.assign_add(global_step, 1, name='increment_global_step')

        '''
        envir = AlphaPuzzle(args)
        advcore = CuriosityRL(learning_rate=1e-3, args=args)
        trainer = a2c.A2CTrainer(envir=envir,
                advcore=advcore,
                tmax=args.batch,
                gamma=config.GAMMA,
                learning_rate=1e-3,
                global_step=global_step)
        '''
        trainer = TrainerMT(args, g, global_step)

        # TODO: Summaries

        saver = tf.train.Saver() # Save everything
        last_time = time.time()
        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            epoch = 0
            accum_epoch = 0
            if args.viewinitckpt:
                advcore.load_pretrain(sess, args.viewinitckpt)
            else:
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
                print('ckpt {}'.format(ckpt))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    accum_epoch = sess.run(global_step)
                    print('Restored!, global_step {}'.format(accum_epoch))
                    if args.continuetrain:
                        accum_epoch += 1
                        epoch = accum_epoch
                else:
                    if args.eval:
                        print('PANIC: --eval is set but checkpoint does not exits')
                        return
            period_loss = 0.0
            period_accuracy = 0
            total_accuracy = 0
            g.finalize() # Prevent accidental changes
            while epoch < total_epoch:
                trainer.train(sess)
                if (not args.eval) and ((epoch + 1) % 1000 == 0 or time.time() - last_time >= 10 * 60 or epoch + 1 == total_epoch):
                    print("Saving checkpoint")
                    fn = saver.save(sess, ckpt_dir+ckpt_prefix, global_step=global_step)
                    print("Saved checkpoint to {}".format(fn))
                    last_time = time.time()
                if (epoch + 1) % 10 == 0:
                    print("Progress {}/{}".format(epoch, total_epoch))
                # print("Epoch {} (Total {}) Done".format(epoch, accum_epoch))
                epoch += 1
                accum_epoch += 1
            trainer.stop() # Must stop before session becomes invalid

if __name__ == '__main__':
    parser = rlargs.get_parser()

    args = parser.parse_args()
    setup_global_variable(args)
    if args.continuetrain:
        if args.samplein:
            print('--continuetrain is incompatible with --samplein')
            exit()
        if args.batching:
            print('--continuetrain is incompatible with --batching')
            exit()
    if -1 in args.actionset:
        args.actionset = [i for i in range(12)]
    if MT_VERBOSE:
        print("Action set {}".format(args.actionset))
    args.total_sample = args.iter * args.threads
    args.total_epoch = args.total_sample / args.samplebatching
    curiosity_main(args)
