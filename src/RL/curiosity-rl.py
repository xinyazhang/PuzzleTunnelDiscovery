'''
    pretrain.py

    Pre-Train the VisionNet and Inverse Model
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.misc import imsave
import matplotlib.animation as animation
import sys
import os
import errno
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
from six.moves import queue,input
from rlreanimator import reanimate
import rlutil
import qtrainer
import ctrainer

MT_VERBOSE = False
# MT_VERBOSE = True
VIEW_CFG = config.VIEW_CFG

COLLIDE_PEN_MAG = 10

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
    return rlutil.get_view_cfg(args)

create_renderer = rlutil.create_renderer

class GroundTruth:
    pass

class AlphaPuzzle(rlenv.IExperienceReplayEnvironment):
    solved_award_mag = 10

    def __init__(self, args, tid, agent_id=-1):
        dumpdir = None
        if args.exploredir is not None:
            assert agent_id >= 0, '--exploredir require AlphaPuzzle constructed with non-negative agent_id'
            dumpdir = '{}/agent-{}/'.format(args.exploredir, agent_id)
            try:
                os.makedirs(dumpdir)
            except OSError as e:
                print("Exc {}".format(e))
                if errno.EEXIST != e.errno:
                    raise
        super(AlphaPuzzle, self).__init__(tmax=args.batch, erep_cap=args.ereplayratio, dumpdir=dumpdir)
        self.fb_cache = None
        self.fb_dirty = True
        r = self.r = create_renderer(args, creating_ctx=False)
        self.istateraw = args.istateraw
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
        self.permutemag = args.permutemag
        self.perturbation = False
        self.dump_id = 0
        self.steps_since_reset = 0
        self.collision_pen_mag = args.collision_pen_mag

    def enable_perturbation(self):
        self.perturbation = True
        self.reset()

    def get_perturbation(self):
        return self.r.perturbation

    def qstate_setter(self, state):
        # print('old {}'.format(self.r.state))
        # print('new {}'.format(state))
        if np.array_equal(self.r.state, state):
            return
        self.r.state = state
        self.fb_dirty = True
        self.steps_since_reset += 1

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

    def peek_act(self, action, pprefix="", start_state=None):
        r = self.r
        start_state = r.state if start_state is None else start_state
        colkey = tuple(start_state.tolist() + [action])
        if colkey in self.collision_cache:
            print("Cache Hit {}".format(colkey))
            nstate, done, ratio = self.collision_cache[colkey]
        else:
            nstate, done, ratio = r.transit_state(start_state,
                    action,
                    self.action_magnitude,
                    self.verify_magnitude)
            if ratio < 1e-4:
                '''
                Disable moving forward if ratio is too small.
                '''
                ratio = 0
                nstate = np.copy(start_state)
        sa = (colkey, (nstate, done, ratio))
        reaching_terminal = r.is_disentangled(nstate)
        reward = 0.0
        reward += pyosr.distance(start_state, nstate) # Reward by translation
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
        return nstate, reward, reaching_terminal, ratio

    def reset(self):
        super(AlphaPuzzle, self).reset()
        if self.perturbation:
            r = self.r
            r.set_perturbation(uw_random.random_state(self.permutemag))
            '''
            Different perturbation has different istate in unit world.
            '''
            self.istate = np.array(r.translate_to_unit_state(self.istateraw), dtype=np.float32)
        self.qstate = self.istate
        self.steps_since_reset = 0

class CuriosityRL(rlenv.IAdvantageCore):
    rgb_shape = None
    dep_shape = None
    fb_dirty = True
    fb_cache = None
    polout = None
    polparams = None
    action_magnitude = None
    verify_magnitude = None

    def __init__(self, learning_rate, args, batch_normalization=None):
        super(CuriosityRL, self).__init__()
        self.view_num, self.views = _get_view_cfg(args)
        w = h = args.res
        self.args = args
        self.batch_normalization = batch_normalization

        self.action_space_dimension = uw_random.DISCRETE_ACTION_NUMBER
        self.action_tensor = tf.placeholder(tf.float32, shape=[None, 1, self.action_space_dimension], name='ActionPh')
        self.rgb_1_tensor = tf.placeholder(tf.float32, shape=[None, self.view_num, w, h, 3], name='Rgb1Ph')
        self.rgb_2_tensor = tf.placeholder(tf.float32, shape=[None, self.view_num, w, h, 3], name='Rgb2Ph')
        self.dep_1_tensor = tf.placeholder(tf.float32, shape=[None, self.view_num, w, h, 1], name='Dep1Ph')
        self.dep_2_tensor = tf.placeholder(tf.float32, shape=[None, self.view_num, w, h, 1], name='Dep2Ph')

        if self.view_num > 1 and not args.sharedmultiview:
            assert False,"Deprecated Code Path, Check Your Arguments"
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
            pm = [pyosr.get_permutation_to_world(self.views, i) for i in range(len(self.views))]
            pm = np.array(pm)
            print(pm)
            with tf.variable_scope(icm.view_scope_name('0')):
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
                        fwhidden=args.fwhidden,
                        permuation_matrix=pm,
                        batch_normalization=batch_normalization)
                self.model.get_inverse_model()

        self.polout, self.polparams, self.polnets = self.create_polnet(args)
        self.valout, self.valparams, self.valnets = self.create_valnet(args)
        if args.curiosity_type == 2:
            self.ratios_tensor = tf.placeholder(tf.float32, shape=[None], name='RatioPh')
        else:
            self.ratios_tensor = None
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
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.loss = None
        # self.refine_vision_and_memory_op = self.optimizer.minimize(var_list = self.curiosity_params)

    def load_pretrained(self, args):
        self.icm.restore(sess, args.viewinitckpt)

    '''
    CAVEAT: MUST use randomized initial value, otherwise ValNet always has the same output.
    '''

    def create_polnet(self, args):
        hidden = args.polhidden + [int(self.action_tensor.shape[-1])]
        return self.model.create_somenet_from_feature(hidden, 'PolNet',
                elu=args.elu,
                lstm=args.lstm,
                initialized_as_zero=False,
                nolu_at_final=True,
                batch_normalization=self.batch_normalization)

    def create_valnet(self, args):
        hidden = args.valhidden + [1]
        return self.model.create_somenet_from_feature(hidden, 'ValNet',
                elu=args.elu,
                lstm=args.lstm,
                initialized_as_zero=False,
                nolu_at_final=True,
                batch_normalization=self.batch_normalization)

    def create_curiosity_net(self, args):
        '''
        Note: we need to train the curiosity model as memory, so tf.losses is used.
        '''
        # curiosity = tf.metrics.mean_squared_error(fwd_feat, self.model.next_featvec)
        if args.curiosity_type == 0:
            return None, None
        if args.curiosity_type == 1:
            fwd_params, fwd_feat = self.model.get_forward_model(args.jointfw)
            curiosity = tf.reduce_mean(tf.squared_difference(fwd_feat, self.model.next_featvec),
                                       axis=[1,2])
            print(">> FWD_FEAT {}".format(fwd_feat.shape))
            print(">> next_featvec {}".format(self.model.next_featvec.shape))
        elif args.curiosity_type == 2:
            # Re-use get_forward_model to generate the ratio prediction
            fwd_params, ratio_out = self.model.get_forward_model(args.jointfw, output_fn=1)
            sigmoid_ratios = tf.sigmoid(ratio_out)
            mean_ratios = tf.reduce_mean(sigmoid_ratios, axis=[1,2])
            print(">> ratios {}".format(self.ratios_tensor.shape))
            print(">> sigmoid_ratios {}".format(sigmoid_ratios.shape))
            print(">> mean_ratios {}".format(mean_ratios.shape))
            curiosity = tf.squared_difference(mean_ratios, self.ratios_tensor)
        else:
            assert False, "Unknown curiosity_type {}".format(args.curiosity_type)
        print(">> curiosity {}".format(curiosity.shape))
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

    def evaluate(self, vstates, sess, tensors, additional_dict=None):
        '''
        Transpose the vstates from [(rgb0,dep0),(rgb1,dep1),...]
        to ([rgb0, rgb1, ...],[dep0, dep1, ...])
        '''
        rgbs = [state[0] for state in vstates]
        deps = [state[1] for state in vstates]
        # NOTE: all nets accepts multiple frames, so we use lists of 1 element for single frame
        dic = {
                self.rgb_1 : rgbs,
                self.dep_1 : deps
              }
        if self.batch_normalization is not None:
            dic[self.batch_normalization] = False
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
            ret = np.asscalar(best)
        else:
            ret = random.randrange(self.action_space_dimension)
        print(pprefix, 'Action best {} chosen {}'.format(best, ret))
        return ret

    def get_artificial_reward(self, envir, sess, state_1, action, state_2, ratio, pprefix=""):
        if self.curiosity is None:
            return 0
        envir.qstate = state_1
        vs1 = envir.vstate
        envir.qstate = state_2
        vs2 = envir.vstate
        return self.get_artificial_from_experience(sess, [vs1,vs2], [action], [ratio], pprefix)[0]
        # TODO: Remove the following piece
        dic = {
                self.action_tensor : [[adist]], # Expand from [12] (A only) to [1,1,12] ([F,V,A])
                self.rgb_1 : [vs1[0]],
                self.dep_1 : [vs1[1]],
                self.rgb_2 : [vs2[0]],
                self.dep_2 : [vs2[1]],
              }
        if self.batch_normalization is not None:
            dic[self.batch_normalization] = False
        '''
        self.curiosity now becomes [None] tensor rather than scalar
        '''
        ret = sess.run(self.curiosity, feed_dict=dic)[0]
        print(pprefix, "AR {}".format(ret))
        print(pprefix, "AR Input adist {} vs1 {} {} vs2 {} {}".format(adist, vs1[0].shape, vs1[1].shape, vs2[0].shape, vs2[1].shape))
        return ret

    def get_artificial_from_experience(self, sess, vstates, actions, ratios, pprefix):
        if self.curiosity is None:
            return np.zeros(shape=(len(actions)))
        '''
        adists_array = []
        for ai in actions:
            adist = np.zeros(shape=(1, self.action_space_dimension),
                    dtype=np.float32)
            adist[0, ai] = 1.0
            adists_array.append(adist)
        '''
        adists_array = rlutil.actions_to_adist_array(actions)
        rgbs = [state[0] for state in vstates]
        deps = [state[1] for state in vstates]
        print('> RGBs {} len: {}'.format(rgbs[0].shape, len(rgbs)))
        dic = {
                self.action_tensor : adists_array,
                self.rgb_1 : rgbs[:-1],
                self.dep_1 : deps[:-1],
                self.rgb_2 : rgbs[1:],
                self.dep_2 : deps[1:],
              }
        if self.batch_normalization is not None:
            dic[self.batch_normalization] = False
        if self.args.curiosity_type == 2:
            dic[self.ratios_tensor] = ratios
        ret = sess.run(self.curiosity, feed_dict=dic)
        return ret

    def train(self, sess, rgb, dep, actions):
        assert False, "Deprecated buggy function"
        dic = {
                self.action_tensor : actions,
                self.rgb_1 : rgb[:-1],
                self.dep_1 : dep[:-1],
                self.rgb_2 : rgb[1:],
                self.dep_2 : dep[1:],
              }
        sess.run(self.refine_vision_and_memory_op, feed_dict=dic)

    def build_loss(self):
        if self.loss is not None:
            return self.loss
        self.inverse_loss = self.model.get_inverse_loss(discrete=True)
        tf.summary.scalar('inverse_loss', self.inverse_loss)
        if self.curiosity is None:
            self.loss = self.inverse_loss
        else:
            self.curiosity_loss = tf.reduce_sum(self.curiosity)
            tf.summary.scalar('curiosity_loss', self.curiosity_loss)
            self.loss = self.inverse_loss + self.curiosity_loss
        return self.loss

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
        if self.view_num == 1 or self.args.sharedmultiview:
            self.model.load_pretrain(sess, viewinitckpt[0])
        else:
            self.model.load_pretrain(sess, viewinitckpt)


class TrainerMT:
    kAsyncTask = 1
    kSyncTask = 2
    kExitTask = -1

    def __init__(self, args, g, global_step, batch_normalization):
        '''
        if len(args.egreedy) != 1 and len(args.egreedy) != args.threads:
            assert False,"--egreedy should have only one argument, or match the number of threads"
        '''
        self.args = args
        self.advcore = CuriosityRL(learning_rate=1e-3, args=args, batch_normalization=batch_normalization)
        self.tfgraph = g
        self.threads = []
        self.taskQ = queue.Queue(args.queuemax)
        self.sessQ = queue.Queue(args.queuemax)
        self.reportQ = queue.Queue(args.queuemax)
        self.bnorm = batch_normalization
        if args.train == 'a2c':
            self.trainer = a2c.A2CTrainer(
                    advcore=self.advcore,
                    tmax=args.batch,
                    gamma=config.GAMMA,
                    # gamma=0.5,
                    learning_rate=1e-6,
                    ckpt_dir=args.ckptdir,
                    global_step=global_step,
                    batch_normalization=self.bnorm,
                    period=args.period,
                    LAMBDA=args.LAMBDA)
        elif args.train == 'QwithGT' or args.qlearning_with_gt:
            self.trainer = qtrainer.QTrainer(
                    advcore=self.advcore,
                    batch=args.batch,
                    learning_rate=1e-4,
                    ckpt_dir=args.ckptdir,
                    period=args.period,
                    global_step=global_step)
        elif args.train == 'curiosity':
            self.trainer = ctrainer.CTrainer(
                    advcore=self.advcore,
                    batch=args.batch,
                    learning_rate=1e-4,
                    ckpt_dir=args.ckptdir,
                    period=args.period,
                    global_step=global_step)
        else:
            assert False, '--train {} not implemented yet'.format(args.train)
        assert not (self.advcore.using_lstm and self.trainer.erep_sample_cap > 0), "CuriosityRL does not support Experience Replay with LSTM"
        for i in range(args.threads):
            thread = threading.Thread(target=self.run_worker, args=(i,g))
            thread.start()
            self.threads.append(thread)

    def run_worker(self, tid, g):
        '''
        IEnvironment is pre-thread object, mainly due to OpenGL context
        '''
        dpy = pyosr.create_display()
        glctx = pyosr.create_gl_context(dpy)

        with g.as_default():
            args = self.args
            '''
            Disable permutation in TID 0
            Completely disable randomized light source for now
            '''
            # if tid != 0:
            thread_local_envirs = [AlphaPuzzle(args, tid, i) for i in range(args.agents)]
            for e in thread_local_envirs[1:]:
                e.enable_perturbation()
            for i in range(1, len(thread_local_envirs)):
                thread_local_envirs[i].egreedy = args.egreedy[i % len(args.egreedy)]
            #else:
                #thread_local_envirs = [AlphaPuzzle(args, tid)]
                # Also disable randomized light position
                # e.r.light_position = uw_random.random_on_sphere(5.0)
            print("[{}] Number of Envirs {}".format(tid, len(thread_local_envirs)))
            while True:
                task = self.taskQ.get()
                if task == self.kExitTask:
                    return 0
                sess = self.sessQ.get()
                '''
                Pickup the envir stochasticly
                '''
                envir = random.choice(thread_local_envirs)
                if not self.args.qlearning_with_gt:
                    print("[{}] Choose Envir with Pertubation {} and egreedy".format(tid, envir.r.perturbation, envir.egreedy))
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

    def load_pretrain(self, sess, pretrained_ckpt):
        self.advcore.load_pretrain(sess, pretrained_ckpt)

class RLVisualizer(object):
    def __init__(self, args, g, global_step):
        self.args = args
        self.dpy = pyosr.create_display()
        self.ctx = pyosr.create_gl_context(self.dpy)
        self.envir = AlphaPuzzle(args, 0)
        self.envir.egreedy = 0.995
        self.advcore = CuriosityRL(learning_rate=1e-3, args=args)
        self.advcore.softmax_policy # Create the tensor
        self.gview = 0 if args.obview < 0 else args.obview
        self.envir.enable_perturbation()

    def attach(self, sess):
        self.sess = sess

class PolicyPlayer(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(PolicyPlayer, self).__init__(args, g, global_step)

    def play(self):
        reanimate(self)

    def __iter__(self):
        envir = self.envir
        sess = self.sess
        advcore = self.advcore
        reaching_terminal = False
        pprefix = "[0] "
        while True:
            rgb,_ = envir.vstate
            yield rgb[self.gview] # First view
            if reaching_terminal:
                print("##########CONGRATS TERMINAL REACHED##########")
                envir.reset()
            policy = advcore.evaluate([envir.vstate], sess, [advcore.softmax_policy])
            policy = policy[0][0]
            action = advcore.make_decision(envir, policy, pprefix)
            print("PolicyPlayer pol {}".format(policy))
            print("PolicyPlayer Action {}".format(action))
            nstate,reward,reaching_terminal,ratio = envir.peek_act(action, pprefix=pprefix)
            envir.qstate = nstate

class QPlayer(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(QPlayer, self).__init__(args, g, global_step)
        if args.permutemag > 0:
            self.envir.enable_perturbation()

    def render(self, envir, state):
        envir.qstate = state
        return envir.vstate

    def play(self):
        if self.args.sampleout:
            self._sample()
        else:
            self._play()

    def _sample(self):
        Q = [] # list of states
        V = [] # list of numpy array of batched values
        args = self.args
        sess = self.sess
        advcore = self.advcore
        envir = self.envir
        assert args.iter % args.batch == 0, "presumably --iter is dividable by --batch"
        for i in range(args.iter/args.batch):
            states = [uw_random.gen_unit_init_state(envir.r) for i in range(args.batch)]
            Q += states
            images = [self.render(envir, state) for state in states]
            batch_rgb = [image[0] for image in images]
            batch_dep = [image[1] for image in images]
            dic = {
                    advcore.rgb_1: batch_rgb,
                    advcore.dep_1: batch_dep,
                  }
            values = sess.run(advcore.value, feed_dict=dic)
            values = np.reshape(values, [-1]) # flatten
            V.append(values)
        Q = np.array(Q)
        V = np.concatenate(V)
        np.savez(args.sampleout, Q=Q, V=V)

    def _play(self):
        reanimate(self)

    def __iter__(self):
        args = self.args
        sess = self.sess
        advcore = self.advcore
        envir = self.envir
        envir.enable_perturbation()
        envir.reset()
        current_value = -1
        TRAJ = []
        while True:
            TRAJ.append(envir.qstate)
            yield envir.vstate[0][args.obview] # Only RGB
            NS = []
            images = []
            # R = []
            T = []
            TAU = []
            state = envir.qstate
            print("> Current State {}".format(state))
            for action in range(uw_random.DISCRETE_ACTION_NUMBER):
                envir.qstate = state # IMPORTANT: Restore the state to unpeeked condition
                nstate, reward, terminal, ratio = envir.peek_act(action)
                envir.qstate = nstate
                NS.append(nstate)
                T.append(terminal)
                TAU.append(ratio)
                image = envir.vstate
                images.append(image)
            batch_rgb = [image[0] for image in images]
            batch_dep = [image[1] for image in images]
            dic = {
                    advcore.rgb_1: batch_rgb,
                    advcore.dep_1: batch_dep,
                  }
            values = sess.run(advcore.value, feed_dict=dic)
            values = np.reshape(values, [-1]) # flatten
            best = np.argmax(values, axis=0)
            print("> Current Values {}".format(values))
            print("> Taking Action {} RATIO {}".format(best, TAU[best]))
            print("> NEXT State {} Value".format(NS[best], values[best]))
            envir.qstate = NS[best]
            should_reset = False
            if current_value > values[best] or TAU[best] == 0.0:
                input("FATAL: Hit Local Maximal! Press Enter to restart")
                should_reset = True
            else:
                current_value = values[best]
            if T[best]:
                input("DONE! Press Enter to restart ")
                should_reset = True
            if should_reset:
                fn = input("Enter the filename to save the trajectory ")
                if fn:
                    TRAJ.append(envir.qstate)
                    TRAJ = np.array(TRAJ)
                    np.savez(fn, TRAJ=TRAJ, SINGLE_PERM=envir.get_perturbation())
                envir.reset()
                current_value = -1
                TRAJ = []

class CuriositySampler(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(CuriositySampler, self).__init__(args, g, global_step)
        assert args.visualize == 'curiosity', '--visualize must be curiosity'
        assert args.curiosity_type == 1, "--curiosity_type should be 1 if --visualize is enabled"
        assert args.sampleout != '', '--sampleout must be enabled for --visualize curiosity'

    def play(self):
        args = self.args
        sess = self.sess
        advcore = self.advcore
        envir = self.envir
        samples= []
        curiosities_by_action = [ [] for i in range(uw_random.DISCRETE_ACTION_NUMBER) ]
        for i in range(args.iter):
            state = uw_random.gen_unit_init_state(envir.r)
            samples.append(state)
            for action in range(uw_random.DISCRETE_ACTION_NUMBER):
                nstate, reward, terminal, ratio = envir.peek_act(action)
                areward = advcore.get_artificial_reward(envir, sess,
                        state, action, nstate, ratio)
                curiosities_by_action[action].append(areward)
        samples = np.array(samples)
        curiosity = np.array(curiosities_by_action)
        np.savez(args.sampleout, Q=samples, C=curiosity)

def create_visualizer(args, g, global_step):
    if args.qlearning_with_gt:
        # assert args.sampleout, "--sampleout is required to store the samples for --qlearning_with_gt"
        assert args.iter > 0, "--iter needs to be specified as the samples to generate"
        # assert False, "Evaluating of Q Learning is not implemented yet"
        player = QPlayer(args, g, global_step)
    elif args.visualize == 'policy':
        return PolicyPlayer(args, g, global_step)
    elif args.visualize == 'curiosity':
        return CuriositySampler(args, g, global_step)
    assert False, '--visualize {} is not implemented yet'.format(args.visualize)

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
        bnorm = tf.placeholder(tf.bool, shape=()) if args.batchnorm else None
        if args.eval:
            player = create_visualizer(args, g, global_step)
        else:
            trainer = TrainerMT(args, g, global_step, batch_normalization=bnorm)

        # TODO: Summaries

        saver = tf.train.Saver() # Save everything
        last_time = time.time()
        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            epoch = 0
            accum_epoch = 0
            if args.viewinitckpt and not args.eval:
                trainer.load_pretrain(sess, args.viewinitckpt)

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
                    print('PANIC: --eval is set but checkpoint does not exist')
                    return

            period_loss = 0.0
            period_accuracy = 0
            total_accuracy = 0
            g.finalize() # Prevent accidental changes
            if args.eval:
                player.attach(sess)
                player.play()
                return
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
    args = rlargs.parse()
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
    print("> Arguments {}".format(args))
    curiosity_main(args)
