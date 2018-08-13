'''
Implement Interfaces in rlenv for Curiosity RL
'''
from __future__ import print_function
# Fundamental packages
import tensorflow as tf
import numpy as np
# Custom RL packages
import pyosr
import rlenv
import rlutil
import uw_random
import icm
# Auxiliary packages
import random
import os
import errno
from cachetools import LRUCache
# Legacy package, ResNet 18 does not need this
import config

def sess_no_hook(sess, tensors, feed_dict):
    if isinstance(sess, tf.Session):
        return sess.run(tensors, feed_dict=feed_dict)
    if isinstance(sess, tf.train.MonitoredSession):
        def step_fn(step_context):
            return step_context.session.run(fetches=tensors, feed_dict=feed_dict)
        return sess.run_step_fn(step_fn)
    assert False, "sess_no_hook: sess is not an instance of tf.Session or tf.MonitoredSession"

class RigidPuzzle(rlenv.IExperienceReplayEnvironment):

    def __init__(self, args, tid, agent_id=-1):
        dumpdir = None
        if args.exploredir is not None:
            assert agent_id >= 0, '--exploredir require RigidPuzzle constructed with non-negative agent_id'
            dumpdir = '{}/agent-{}/'.format(args.exploredir, agent_id)
            try:
                os.makedirs(dumpdir)
            except OSError as e:
                print("Exc {}".format(e))
                if errno.EEXIST != e.errno:
                    raise
        super(RigidPuzzle, self).__init__(tmax=args.batch, erep_cap=args.ereplayratio, dumpdir=dumpdir)
        self.args = args
        self.fb_cache = None
        self.fb_dirty = True
        r = self.r = rlutil.create_renderer(args, creating_ctx=False)
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
            self.egreedy = args.egreedy[tid % len(args.egreedy)]
        self.permutemag = args.permutemag
        self.perturbation = False
        self.dump_id = 0
        self.steps_since_reset = 0
        self.PEN = args.PEN
        self.REW = args.REW
        if args.msi_file:
            dic = np.load(args.msi_file)
            self.traj_s = dic['TRAJ_S']
            assert self.traj_s[0].all() == self.istate.all(), "--mis_file does not match istate"
            self.traj_a = dic['TRAJ_A']
            self.msi_index = 0
        else:
            self.traj_a = None
            self.msi_index = None
        # lstm_barn: a cookie property to track the lstm states
        self.lstm_barn = None

    def enable_perturbation(self, manual_p=None):
        self.perturbation = True
        self.manual_p = manual_p
        self.reset()

    def disable_perturbation(self):
        self.perturbation = False
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
        # reward += pyosr.distance(start_state, nstate) # Reward by translation
        reward += self.REW if reaching_terminal is True else 0.0 # Large Reward for solution
        if not done:
            '''
            Special handling of collision
            '''
            if ratio == 0.0:
                reward += self.PEN
                if self.args.die:
                    reaching_terminal = True
            self.collision_cache.update([sa])
        rgb_1, dep_1 = self.vstate
        self.state = nstate
        rgb_2, dep_2 = self.vstate
        print("{}New state {} ratio {} terminal {} reward {}".format(pprefix, nstate, ratio, reaching_terminal, reward))
        return nstate, reward, reaching_terminal, ratio

    def reset(self):
        super(RigidPuzzle, self).reset()
        if self.perturbation:
            r = self.r
            if self.manual_p is None:
                r.set_perturbation(uw_random.random_state(self.permutemag))
            else:
                r.set_perturbation(self.manual_p)
            '''
            Different perturbation has different istate in unit world.
            '''
            self.istate = np.array(r.translate_to_unit_state(self.istateraw), dtype=np.float32)
        self.qstate = self.istate
        self.steps_since_reset = 0
        if self.msi_index is not None:
            self.msi_index = 0

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
        self.view_num, self.views = rlutil.get_view_cfg(args)
        w = h = args.res
        self.args = args
        self.batch_normalization = batch_normalization
        self.nn_vars = dict()

        self.action_space_dimension = uw_random.DISCRETE_ACTION_NUMBER

        batch_size = None if args.EXPLICIT_BATCH_SIZE < 0 else args.EXPLICIT_BATCH_SIZE
        self.batch_size = batch_size

        common_shape = [batch_size, self.view_num, w, h]
        self.action_tensor = tf.placeholder(tf.float32, shape=[batch_size, 1, self.action_space_dimension], name='ActionPh')
        self.rgb_1_tensor = tf.placeholder(tf.float32, shape=common_shape+[3], name='Rgb1Ph')
        self.rgb_2_tensor = tf.placeholder(tf.float32, shape=common_shape+[3], name='Rgb2Ph')
        self.dep_1_tensor = tf.placeholder(tf.float32, shape=common_shape+[1], name='Dep1Ph')
        self.dep_2_tensor = tf.placeholder(tf.float32, shape=common_shape+[1], name='Dep2Ph')

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
            pm = np.array(pm) if args.view < 0 else None
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
                self.model.create_pretrain_saver()

        self.polout, self.polparams, self.polnets = self.create_polnet(args)
        # self.polout = tf.Variable(np.zeros(dtype=np.float32, shape=(54,1,self.action_space_dimension)))
        if 'dqn' not in args.train:
            self.valout, self.valparams, self.valnets = self.create_valnet(args)
        else:
            self.valout = tf.constant(0.0)
            self.valparams = []
            self.valnets = None
        if args.curiosity_type == 2:
            self.ratios_tensor = tf.placeholder(tf.float32, shape=[batch_size], name='RatioPh')
        self.curiosity, self.curiosity_params = self.create_curiosity_net(args)
        print('Curiosity Params: {}'.format(self.curiosity_params))

        self.using_lstm = args.lstm
        if self.using_lstm:
            self.lstm_states_in, self.lstm_len, self.lstm_states_out = self.model.acquire_lstm_io('LSTM')
            # batching is madantory, although our batch size is 1
            self.current_lstm = tf.contrib.rnn.LSTMStateTuple(
                    np.zeros([1, self.model.lstmsize], dtype=np.float32),
                    np.zeros([1, self.model.lstmsize], dtype=np.float32))
            print("[LSTM] {}".format(self.lstm_states_in.c))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.loss = None
        # self.refine_vision_and_memory_op = self.optimizer.minimize(var_list = self.curiosity_params)

    def load_pretrained(self, sess, args):
        self.icm.restore(sess, args.viewinitckpt)

    '''
    CAVEAT: MUST use randomized initial value, otherwise ValNet always has the same output.
    '''

    def create_polnet(self, args):
        hidden = args.polhidden + [int(self.action_tensor.shape[-1])]
        return self.model.create_somenet_from_feature(hidden, 'PolNet',
                elu=args.elu,
                lstm=args.lstm,
                initialized_as_zero=True if 'zero_weights' in args.debug_flags else False,
                nolu_at_final=True,
                batch_normalization=self.batch_normalization)

    def create_valnet(self, args):
        hidden = args.valhidden + [1]
        return self.model.create_somenet_from_feature(hidden, 'ValNet',
                elu=args.elu,
                lstm=args.lstm,
                initialized_as_zero=True if 'zero_weights' in args.debug_flags else False,
                nolu_at_final=True,
                batch_normalization=self.batch_normalization)

    def create_curiosity_net(self, args):
        '''
        Note: we need to train the curiosity model as memory, so tf.losses is used.
        '''
        # curiosity = tf.metrics.mean_squared_error(fwd_feat, self.model.next_featvec)
        if args.curiosity_type == 0:
            return None, []
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
    def softmax_policy(self):
        if self._softmax_policy_tensor is None:
            actionset = self.args.actionset
            asize = len(actionset)
            extractor = np.zeros((self.action_space_dimension, asize), dtype=np.float32)
            for i, a in enumerate(actionset):
                extractor[a, i] = 1.0
            adaptor = np.transpose(extractor)
            assert len(self.policy.shape) == 3, 'shape of policy ({}) does not follow (B,V,A)'.format(self.policy.shape)
            compact = tf.tensordot(self.policy, extractor, [[2], [0]])
            assert int(compact.shape[-1]) == asize, 'softmax_policy compact shape assertion failed'
            compact_softmax = tf.nn.softmax(compact)
            self._softmax_policy_tensor = tf.tensordot(compact_softmax, adaptor, [[2],[0]])
            assert int(self._softmax_policy_tensor.shape[-1]) == self.action_space_dimension, "softmax policy shape failed"
        return self._softmax_policy_tensor
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
            return sess_no_hook(sess, tensors, feed_dict=dic)
        assert False, "TODO: replace sess.run with run_step_fn"
        # print(self.lstm_states_in.c)
        # print(self.current_lstm)
        dic.update({
            self.lstm_states_in.c : self.current_lstm.c,
            self.lstm_states_in.h : self.current_lstm.h,
            self.lstm_len : 1
                   })
        # FIXME: verify ret[-1] is a tuple
        ret = sess_no_hook(tensors + [self.lstm_states_out], feed_dict=dic)
        self.lstm_cache = ret[-1]
        return ret[:-1]

    def make_decision(self, envir, subspace_policy_dist, pprefix=''):
        if envir.msi_index is not None:
            ret = np.asscalar(envir.traj_a[envir.msi_index])
            envir.msi_index += 1
            return ret
        actionset = self.args.actionset
        best = np.argmax(subspace_policy_dist, axis=-1)
        if random.random() < envir.egreedy:
            ret = np.asscalar(best)
        else:
            ret = np.random.choice(actionset)
        # print(pprefix, 'Action best index {} ({}) chosen index {} ({})'.format(best, actionset[best], ret, actionset[ret]))
        print(pprefix, 'Action best {} chosen {}'.format(best, ret))
        assert ret in actionset, 'san check failed: make_decision return unselected action'
        return ret

    def get_artificial_reward(self, envir, sess, state_1, action, state_2, ratio, pprefix=""):
        if self.curiosity is None:
            return 0
        envir.qstate = state_1
        vs1 = envir.vstate
        envir.qstate = state_2
        vs2 = envir.vstate
        return self.get_artificial_from_experience(sess, [vs1,vs2], [action], [ratio], pprefix)[0]

    def get_artificial_from_experience(self, sess, vstates, actions, ratios, pprefix=''):
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
        adists_array = rlutil.actions_to_adist_array(actions, dim=self.action_space_dimension)
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
        ret = sess_no_hook(sess, self.curiosity, feed_dict=dic)
        return ret * self.args.curiosity_factor

    def train(self, sess, rgb, dep, actions):
        assert False, "Deprecated buggy function"
        dic = {
                self.action_tensor : actions,
                self.rgb_1 : rgb[:-1],
                self.dep_1 : dep[:-1],
                self.rgb_2 : rgb[1:],
                self.dep_2 : dep[1:],
              }
        # Note: do NOT use sess_no_hook, since this is training the network
        sess.run(self.refine_vision_and_memory_op, feed_dict=dic)

    def build_loss(self):
        if self.loss is not None:
            return self.loss
        self.inverse_loss = self.model.get_inverse_loss(discrete=True)
        # self.inverse_loss = tf.constant(-1, dtype=tf.float32) # Disabled
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

'''
Dummpy Advcore
'''
class FeatureVectorOverfitRL(object):
    def __init__(self, learning_rate, args, batch_normalization=None):
        pass

def create_advcore(learning_rate, args, batch_normalization):
    if 'a2c_overfit_from_fv' == args.train:
        return FeatureVectorOverfitRL(learning_rate=learning_rate, args=args, batch_normalization=batch_normalization)
    return CuriosityRL(learning_rate=learning_rate, args=args, batch_normalization=batch_normalization)
