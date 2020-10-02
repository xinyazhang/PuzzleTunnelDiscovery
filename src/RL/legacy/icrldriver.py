# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import pyosr
import numpy as np
import tensorflow as tf
import vision
import config
import random
from math import sqrt,pi,sin,cos
import icm

class ICRLDriver:
    '''
    ICRLDriver: Intrinsic Curiosity Reinforcement Learning Driver

    Changes over original RLDriver
        1. Assumes AdamOptimizer rather than RMSProps
        2. Synchronized (aka A2C)
    '''
    renderer = None
    master = None
    sv_rgb_net = None
    sv_depth_net = None
    mv_net = None
    decision_net_args = []
    value_net_args = []
    sync_op_group = None
    mags = config.STATE_TRANSITION_DEFAULT_MAGS
    deltas = config.STATE_TRANSITION_DEFAULT_DELTAS
    max_iteration_per_epoch = config.MAX_ITERATION_PER_EPOCH
    a3c_local_t = config.A3C_LOCAL_T
    a3c_gamma = config.RL_GAMMA
    a3c_entropy_beta = config.ENTROPY_BETA
    action_size = 0
    total_loss = None
    grads_applier = None
    grads_apply_op = None
    epsilon = 0.8 # argument for epsilon-greedy policy
    worker_thread_index = -1
    init_state = None
    continuous_policy_loss = False
    VERIFICATION_DELTA = 0.0125 / 64
    verbose_training = False

    '''
        models: files name of [model, robot], or [model], or [model, None]
        view_config: array of (angle, number cameras)
        svconfdict: network configuration dict for Single View (SV) CNN, e.g. config.SV_VISCFG
        mvconfdict: network configuration dict for Multiview View (MV) CNN
        sv_sqfeatnum: Squared Root of feature numbers for SV CNN
    '''
    def __init__(self,
            models,
            init_state,
            view_config,
            svconfdict,
            mvconfdict,
            output_number = 3 + 3,
            sv_sqfeatnum = 16,
            mv_featnum = 256,
            use_rgb = True, # Obsoleted, always True
            master_driver = None,
            grads_applier = None
            worker_thread_index = -1,
            continuous_policy_loss = False):
        self.init_state = init_state
        self.worker_thread_index = worker_thread_index
        if grads_applier is not None:
            self.grads_applier = grads_applier
        else:
            self.grads_applier = tf.AdamOptimizer()
        self.action_size = output_number
        self.continuous_policy_loss = continuous_policy_loss
        self.renderer = pyosr.Renderer()
        r = self.renderer
        r.pbufferWidth = config.DEFAULT_RES
        r.pbufferHeight = config.DEFAULT_RES
        if master_driver is None:
            r.setup()
            r.loadModelFromFile(models[0])
            if len(models) > 1 and models[1] is not None:
                r.loadRobotFromFile(models[1])
                r.state = np.array(init_state, dtype=np.float32)
            print('robot loaded')
            r.scaleToUnit()
            r.angleModel(0.0, 0.0)
            r.default_depth = 0.0
        else:
            r.setupFrom(master_driver.renderer)
            r.default_depth = master_driver.renderer.default_depth
            r.state = np.array(init_state, dtype=np.float32)
        self.master = master_driver

        view_array = []
        for angle,ncam in view_config:
            view_array += [ [angle,float(i)] for i in np.arange(0.0, 360.0, 360.0/float(ncam)) ]
        r.views = np.array(view_array, dtype=np.float32)
        view_num = len(view_array)

        w = r.pbufferWidth
        h = r.pbufferHeight

        # Action taken
        self.a3c_batch_a_tensor = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.rgb1_tensor = tf.placeholder(tf.float32, shape=[None, view_num, w, h, 3])
        self.rgb2_tensor = tf.placeholder(tf.float32, shape=[None, view_num, w, h, 3])
        self.dep1_tensor = tf.placeholder(tf.float32, shape=[None, view_num, w, h, 1])
        self.dep2_tensor = tf.placeholder(tf.float32, shape=[None, view_num, w, h, 1])
        self.icm = icm.IntrinsicCuriosityModule(
                self.a3c_batch_a_tensor,
                self.rgb1_tensor, self.dep1_tensor,
                self.rgb2_tensor, self.dep2_tensor,
                svconfdict, mvconfdict,
                mv_featnum)

        self.policy_applier = vision.ConvApplier(None, config.POLICY_HIDDEN_LAYERS + [output_number])
        params,out = self.policy_applier.infer(self.icm.cur_featvec)
        self.policy_fc_params = params
        self.policy_params = self.cur_nn_params + params
        self.policy_out = out
        print('RLDRIVER POLICY PARAMS {}'.format(self.policy_out))
        print('RLDRIVER POLICY OUTPUT {}'.format(self.policy_out))

        self.value_applier = vision.ConvApplier(None, config.VALUE_HIDDEN_LAYERS + [1])
        params,out = self.value_applier.infer(self.icm.cur_featvec)
        self.value_fc_params = params
        self.value_params = self.cur_nn_params + params
        self.value_out = out
        print('RLDRIVER VALUE PARAMS {}'.format(self.value_out))
        print('RLDRIVER VALUE OUTPUT {}'.format(self.value_out))

    def get_nn_params(self):
        args = []
        args.append(self.icm.get_nn_params())
        if self.sv_rgb_net is not None:
            args.append(self.sv_rgb_net.get_nn_args())
        args.append(self.mv_net.get_nn_args())
        args.append(self.decision_net_args)
        args.append(self.value_net_args)
        return sum(args, []) # Concatenate (+) all element in args, which is a list of list.

    def get_sync_from_master_op(self):
        if self.sync_op_group is not None:
            return self.sync_op_group
        master_args = self.master.get_nn_args()
        self_args = self.get_nn_args()
        # print('master_args {}'.format(master_args))
        # print('self_args {}'.format(self_args))
        sync_ops = []

        for src,dst in zip(master_args, self_args):
            sync_op = tf.assign(dst, src)
            sync_ops.append(sync_op)

        self.sync_op_group = tf.group(*sync_ops)
        return self.sync_op_group

    def sync_from_master(self, sess):
        if self.master is not None:
            sess.run(self.get_sync_from_master_op())

    def is_end(self, state):
        # return np.linalg.norm(state[0:3]) > 1.0
        return self.renderer.is_disentangled(state)

    @staticmethod
    def get_action_magnitude(action):
        return np.linalg.norm(action[0:3]) + 0.5 * np.linalg.norm(action[3:6])

    def get_reward(self, action):
        r = self.renderer
        if self.continuous_policy_loss:
            nstate, done, ratio = r.transit_state_by(r.state,
                    action[0:3],
                    action[3:6],
                    self.VERIFICATION_DELTA)
        else:
            nstate, done, ratio = r.transit_state(r.state, action,
                    config.MAGNITUDES, config.STATE_CHECK_DELTAS)
        reward = 0.0
        '''
        if not done and ratio > 0.0:
            reward = 0.002
        elif ratio == 0.0:
            reward = -0.002
        '''
        if ratio == 0.0:
            if self.continuous_policy_loss:
                reward = -get_action_magnitude(action)
            else:
                reward = -0.002
        reaching_end = self.is_end(nstate)
        reward = 1.0 if reaching_end else reward
        return nstate, reward, reaching_end

    def make_decision(self, policy, sess):
        '''
        Implement epsilon-greedy policy here

        TODO: incorporate RRT here?
        '''
        # return 5
        greedy = random.random()
        if greedy < self.epsilon:
            '''
            Based on policy output
            '''
            if self.continuous_policy_loss:
                return policy
            else:
                return np.argmax(policy)

            '''
            Based on value output
            r = self.renderer
            state_backup = r.state
            max_value = 0.0
            action_pick = None
            for action in range(self.action_size):
                nstate, _, _= r.transit_state(state_backup, action,
                        config.MAGNITUDES, config.STATE_CHECK_DELTAS)
                r.state = nstate
                value = self.evaluate(sess, targets=[self.value], keep_images=False)
                if action_pick is None or value > max_value:
                    action_pick = action
                    max_value = value
            r.state = state_backup
            return action_pick
            '''
        else:
            if self.continuous_policy_loss:
                return uw_random.random_continuous_action(self.stepping_size)
            else:
                return random.randrange(len(policy))

    def osrender(self):
        r = self.renderer
        if self.sv_rgb_net:
            r.render_mvrgbd()
            img = r.mvrgb.reshape(self.sv_rgb_shape)
            # print('img shape {}'.format(img.shape))
            dep = r.mvdepth.reshape(self.sv_depth_shape)
        else:
            img = None
            dep = r.render_mvdepth_to_buffer().reshape(self.sv_depth_shape)

        return [img, dep]

    def evaluate(self, sess, targets=None, keep_images=True):
        if targets is None:
            targets = [self.final, self.value]
        r = self.renderer
        [img, dep] = self.osrender()
        if self.sv_rgb_net:
            input_dict = {self.sv_rgb_net.input_tensor : img,
                    self.sv_depth_net.input_tensor : dep }
        else:
            input_dict = { self.sv_depth_net.input_tensor : dep }
        if keep_images:
            return sess.run(targets, feed_dict=input_dict) + [img, dep]
        else:
            return sess.run(targets, feed_dict=input_dict)

    def act(self, sess):
        r = self.renderer
        reaching_terminal = False
        while not reaching_terminal:
            policy, value, img, dep = self.evaluate(sess)
            policy = policy.reshape(self.action_size)
            action = self.make_decision(policy, sess)
            nstate,reward,reaching_terminal = self.get_reward(action)

    def train_from_gt(self, sess, keys, cont_tr, cont_rot, gtvalues):
        '''
        Train the NN according to the known ground truth.
        '''
        r = self.renderer

        epsilon_bak = self.epsilon # Disable exploration when training from GT
        r.state = r.translate_to_unit_state(keys[0])
        nactions = cont_tr.shape[0]
        sync_from_master(sess)

        states = []
        actions = []
        rewards = []
        values = []

        for i in range(nactions):
            states.append(self.osrender())
            action = np.concatenate((cont_tr[i], cont_rot[i]))
            actions.append(action)
            values.append(gtvalues[i])
            nstate,reward,reaching_terminal = self.get_reward(action)
            if self.verbose_training:
                print('current state: {}'.format(r.state))
                print('next state: {}'.format(nstate))
                print('\tReaching terminal?: {}'.format(reaching_terminal))
                print('\tDisentangled?: {}'.format(r.is_disentangled(nstate)))
            rewards.append(reward)
            r.state = nstate
            if reaching_terminal:
                break

        # print(len(states))
        nstates = len(states)
        for base in range(0, nstates, self.a3c_local_t):
            end = min(nstates, base + self.a3c_local_t)
            print("{}:{}".format(base, end))
            self.apply_grads_a3c(sess,
                    actions[base:end],
                    states[base:end],
                    rewards[base:end],
                    values[base:end],
                    reaching_terminal)

        sync_from_master(sess)
        self.epsilon = epsilon_bak # Enable exploration again

    def train_a3c(self, sess):
        r = self.renderer

        for it in range(self.max_iteration_per_epoch):
            sync_from_master(sess)
            reaching_terminal = False
            states = []
            actions = []
            rewards = []
            values = []
            start_at = r.state
            for i in range(self.a3c_local_t):
                policy, value, img, dep = self.evaluate(sess)
                policy = policy.reshape(self.action_size)
                value = np.asscalar(value)
                # print("policy {}".format(policy))
                # print("value {}".format(value))
                action = self.make_decision(policy, sess)

                states.append([img, dep])
                actions.append(action)
                values.append(value)

                nstate,reward,reaching_terminal = self.get_reward(action)
                rewards.append(reward)
                r.state = nstate
                if reaching_terminal:
                    break;
            if self.verbose_training:
                end_at = r.state
                r.state = start_at
                policy_old, value_old, _, _ = self.evaluate(sess)
                r.state = end_at

            self.apply_grads_a3c(sess, actions, states, rewards, values, reaching_terminal)

            if self.verbose_training:
                policy_new, value_new, _, _ = self.evaluate(sess)
                print('[{}] state {}'.format(self.worker_thread_index, start_at))
                print('[{}] rewards {}'.format(self.worker_thread_index, rewards))
                print('[{}] value old {}'.format(self.worker_thread_index, value_old))
                print('[{}] value new {}'.format(self.worker_thread_index, value_new))
                print('[{}] policy old {}'.format(self.worker_thread_index, policy_old))
                print('[{}] policy new {}'.format(self.worker_thread_index, policy_new))

    def get_total_loss(self):
        if self.total_loss is not None:
            return self.total_loss

        # Temporal Difference
        self.a3c_batch_td_tensor = tf.placeholder(tf.float32, shape=[None])

        # Log policy, clipped to prevent NaN
        if not self.continuous_policy_loss:
            log_policy = tf.log(tf.clip_by_value(self.final, 1e-20, 1.0))
            entropy = -tf.reduce_sum(self.final * log_policy, reduction_indices=[1])
            policy_loss = -tf.reduce_sum(tf.reduce_sum(tf.multiply(log_policy,
                self.a3c_batch_a_tensor), reduction_indices=1) * self.a3c_batch_td_tensor + entropy
                * self.a3c_entropy_beta)
        else:
            '''
            diff = self.a3c_batch_a_tensor - self.final
            policy_loss = -tf.reduce_sum(tf.reduce_sum(diff * diff, axis=1) * self.a3c_batch_td_tensor))
            '''
            policy_loss = tf.nn.l2_loss(self.final - self.a3c_batch_a_tensor)

        self.a3c_batch_V_tensor = tf.placeholder(tf.float32, shape=[None])
        value_loss = tf.nn.l2_loss(self.a3c_batch_V_tensor - self.value)
        self.total_loss = policy_loss + value_loss
        return self.total_loss

    def get_apply_grads_op(self):
        if self.grads_apply_op is not None:
            return self.grads_apply_op
        print("self.master {}".format(self.master))
        self.grads = tf.gradients(self.get_total_loss(), self.get_nn_args(),
                gate_gradients=False, aggregation_method=None,
                colocate_gradients_with_ops=False)
        self.grads_apply_op = self.grads_applier.apply_gradients(self.master.get_nn_args(),
                self.grads)
        return self.grads_apply_op

    def restart_epoch(self):
        '''
        Generate a valid random configuration

        TODO: non-uniform distribution
        '''
        '''
        self.renderer.state = self.init_state
        return
        '''
        r = self.renderer
        while True:
            tr = np.random.rand(3) * 2.0 - 1.0
            u1,u2,u3 = np.random.rand(3)
            quat = [sqrt(1-u1)*sin(2*pi*u2),
                    sqrt(1-u1)*cos(2*pi*u2),
                    sqrt(u1)*sin(2*pi*u3),
                    sqrt(u1)*cos(2*pi*u3)]
            part1 = np.array(tr, dtype=np.float32)
            part2 = np.array(quat, dtype=np.float32)
            r.state = np.concatenate((part1, part2))
            if r.is_disentangled(r.state):
                continue
            if r.is_valid_state(r.state):
                break


    def apply_grads_a3c(self, sess, actions, states, rewards, values, reaching_terminal):
        '''
        Calculate the grads w.r.t. the loss from policy delta and value delta
        Note: please ensure self.renderer.state is the last state, otherwise R
              would be incorrect.
        '''
        V = 0.0
        if not reaching_terminal:
            V = np.asscalar(self.evaluate(sess, targets=[self.value])[0])

        states.reverse()
        actions.reverse()
        rewards.reverse()
        values.reverse()

        batch_rgba = []
        batch_depth = []
        batch_a = []
        batch_td = []
        batch_V = []
        '''
        Calculate the per-step "true" value for current iteration
        '''
        '''
        print('[{}] R start with {}'.format(self.worker_thread_index, R))
        '''
        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            if self.continuous_policy_loss:
                # Linear decay
                V = min(Vi, -ri + V)
            else:
                V = ri + self.a3c_gamma * V
            td = V - Vi
            a = ai

            if self.sv_rgb_net:
                img = np.squeeze(si[0], [0])
                dep = np.squeeze(si[1], axis=0)
                # print('dep {}'.format(dep.shape))
            else:
                img = None
                dep = np.squeeze(si[0], [0])
            batch_rgba.append(img)
            batch_depth.append(dep)
            batch_a.append(a)
            batch_td.append(td)
            batch_V.append(V)
        if self.verbose_training:
            print('[{}] batch_a[0] {}'.format(self.worker_thread_index, batch_a[0]))
            print('[{}] batch_V {}'.format(self.worker_thread_index, batch_R))
        '''
        TODO: reverse batch_* if using LSTM
        '''
        cur_learning_rate = 1.0 * (10.0 ** -3.0) # FIXME
        grads_applier = self.get_apply_grads_op()
        if self.sv_rgb_net is not None:
            sess.run(grads_applier,
                feed_dict={
                    self.sv_rgb_net.input_tensor: batch_rgba,
                    self.sv_depth_net.input_tensor: batch_depth,
                    self.a3c_batch_a_tensor: batch_a,
                    self.a3c_batch_td_tensor: batch_td,
                    self.a3c_batch_V_tensor: batch_V,
                    self.learning_rate_input: cur_learning_rate} )
        else:
            sess.run(grads_applier,
                feed_dict={
                    self.sv_depth_net.input_tensor: batch_depth,
                    self.a3c_batch_a_tensor: batch_a,
                    self.a3c_batch_td_tensor: batch_td,
                    self.a3c_batch_V_tensor: batch_V,
                    self.learning_rate_input: cur_learning_rate} )
