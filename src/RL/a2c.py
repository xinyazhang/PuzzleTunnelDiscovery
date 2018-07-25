from __future__ import print_function
import tensorflow as tf
import rlenv
import numpy as np
from collections import deque
import itertools
# imsave for Debug
from scipy.misc import imsave
import threading
from six.moves import queue, input
import copy
import rlutil
import curiosity

class RLSample(object):
    def __init__(self, advcore, envir, sess, is_terminal=False):
        # Capture Current Frame
        self.qstate = envir.qstate
        self.perturbation = envir.get_perturbation()
        self.vstate = envir.vstate
        if is_terminal:
            self.policy = None
            self.value = 0.0
        else:
            # Sample Pi and V
            policy, value = advcore.evaluate([envir.vstate], sess, [advcore.softmax_policy, advcore.value])
            self.policy = policy[0][0] # Policy View from first qstate and first view
            self.value = np.asscalar(value) # value[0][0][0]

    '''
        Side effects:
            1. envir.qstate is changed according to the evaluated policy function
            2. self.advcore.lstm_state is changed
    '''
    def proceed(self, advcore, envir, sess):
        # Sample Action
        self.action_index = advcore.make_decision(envir, self.policy)
        # Preview Next frame
        self.nstate, self.true_reward, self.reaching_terminal, self.ratio = envir.peek_act(self.action_index)
        # Artificial Reward
        self.artificial_reward = advcore.get_artificial_reward(envir, sess,
                envir.qstate, self.action_index, self.nstate, self.ratio)
        self.combined_reward = self.true_reward + self.artificial_reward
        # Side Effects: change qstate
        envir.qstate = self.nstate
        # Side Effects: Maintain LSTM
        lstm_next = copy.deepcopy(advcore.get_lstm()) # Get the output of current frame
        advcore.set_lstm(lstm_next) # AdvCore next frame

class A2CSampler(object):

    def __init__(self,
                 advcore,
                 tmax):
        self.advcore = advcore
        self.a2c_tmax = tmax

    '''
    Sample one in a mini-batch

    Side effects: self.advcore.lstm_state is changed
    '''
    def _sample_one(self, envir, sess):
        sam = RLSample(self.advcore, envir, sess)
        sam.proceed(self.advcore, envir, sess)
        return sam

    def sample_minibatch(self, envir, sess, tid=None, tmax=-1):
        if tmax < 0:
            tmax = self.a2c_tmax
        advcore = self.advcore
        samples = []
        # LSTM is also tracked by Envir, since it's derived by vstate
        # FIXME: Initialize envir.lstm_barn somewhere else.
        advcore.set_lstm(envir.lstm_barn)
        for i in range(tmax):
            s = self._sample_one(envir, sess)
            samples.append(s)
            if s.reaching_terminal:
                break
        reaching_terminal = samples[-1].reaching_terminal
        envir.lstm_barn = copy.deepcopy(advcore.get_lstm())
        final = RLSample(self.advcore, envir, sess, is_terminal=reaching_terminal)
        if reaching_terminal:
            envir.reset()
        return (samples, final)


# TODO: A2CSampler should be owned rather than subclassed by A2CTrainer
class A2CTrainer(A2CSampler):
    a2c_tmax = None
    optimizer = None
    loss = None
    verbose_training = False

    def __init__(self,
                 advcore,
                 tmax,
                 gamma,
                 learning_rate,
                 ckpt_dir,
                 global_step=None,
                 entropy_beta=0.01,
                 debug=True,
                 batch_normalization=None,
                 period=1,
                 total_number_of_replicas=None,
                 LAMBDA=0.5,
                 train_everything=False
                ):
        super(A2CTrainer, self).__init__(
                advcore=advcore,
                tmax=tmax)
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.debug = debug
        self.action_space_dimension = int(advcore.policy.shape[-1])
        self.batch_normalization = batch_normalization
        '''
        Create the optimizers to train the AdvCore
        '''
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if period > 1:
            self.optimizer = tf.train.SyncReplicasOptimizer(self.optimizer,
                    replicas_to_aggregate=period,
                    total_num_replicas=total_number_of_replicas)
        LAMBDA_1 = 1
        LAMBDA_2 = 10
        self.loss = LAMBDA_1 * self.build_loss(advcore)
        print("self.loss 1 {}".format(self.loss))
        tf.summary.scalar('a2c_loss', self.loss)
        self.loss += LAMBDA_2 * advcore.build_loss()
        print("self.loss 2 {}".format(self.loss))
        if train_everything is False:
            '''
            Approach 1: With --viewinitckpt, do not train Vision since we don't
                        have reliable GT from RL procedure
            '''
            var_list = advcore.policy_params
            var_list += advcore.value_params
            if advcore.curiosity_params is not None: # guard for --curiosity_type 0
                var_list += advcore.curiosity_params
            var_list += advcore.lstm_params
            self.train_op = self.optimizer.minimize(self.loss,
                    global_step=global_step,
                    var_list=var_list)
        else:
            '''
            Approach 2: Train everything
            '''
            if batch_normalization is not None:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
            else:
                self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
        if ckpt_dir is not None:
            self.summary_op = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(ckpt_dir + '/summary', tf.get_default_graph())
        else:
            self.summary_op = None
            self.train_writer = None
        self.global_step = global_step
        self.dbg_sample_peek = 0

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    '''
    Private: Return A2C Loss

    Side effect: self.rl_params were set
    '''
    def build_loss(self, advcore):
        if self.loss is not None:
            return self.loss

        '''
        Input tensor of Ground Truth from Environment
        '''
        '''
        self.Adist_tensor = tf.placeholder(tf.float32,
                shape=[None, 1, self.action_space_dimension],
                name='ADistPh')
        '''
        self.Adist_tensor = advcore.action_tensor

        self.TD_tensor = tf.placeholder(tf.float32, shape=[None], name='TDPh')
        self.V_tensor = tf.placeholder(tf.float32, shape=[None], name='VPh')
        # self.TD_tensor = tf.placeholder(tf.float32, shape=[None])
        # self.V_tensor = tf.placeholder(tf.float32, shape=[None])

        '''
        # Old buggy policy loss
        policy = advcore.softmax_policy
        log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
        # cond_prob = tf.reduce_sum(policy * self.Adist_tensor, axis=1)
        rindices = [i for i in range(1, len(log_policy.shape))]
        criticism = self.TD_tensor
        self.print('rindices {}'.format(rindices))
        action_entropy = tf.reduce_sum(tf.multiply(log_policy, self.Adist_tensor),
                reduction_indices=rindices)
        entropy = -tf.reduce_sum(policy * log_policy, reduction_indices=rindices)
        self.print('action_entropy shape {}'.format(action_entropy.shape))

        # Why do we add entropy to our loss?
        # policy_loss_per_step = tf.reduce_sum(action_entropy * self.TD_tensor) + entropy * self.entropy_beta
        policy_loss_per_step = tf.reduce_sum(action_entropy * self.TD_tensor)
        policy_loss = -tf.reduce_sum(policy_loss_per_step)
        flattened_value = tf.reshape(advcore.value, [-1])
        value_loss = tf.nn.l2_loss(self.V_tensor - flattened_value)
        '''

        '''
        New policy loss
        '''
        # Need V as critic
        # advcore.value's shape is (B,V,1)
        flattened_value = tf.reshape(advcore.value, [-1])
        self.flattened_value = flattened_value
        # Pick out the sampled action from policy output
        # Shape: (B,V,A)
        policy = tf.multiply(advcore.softmax_policy, self.Adist_tensor)
        assert advcore.softmax_policy.shape.as_list() == self.Adist_tensor.shape.as_list(), "shape match failure: advcore.softmax_policy {} Adist_tensor {}".format(advcore.softmax_policy.shape, self.Adist_tensor.shape)
        policy = tf.reduce_sum(policy, axis=[1,2]) # Shape: (B) afterwards
        log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))

        # TD_tensor input is the same as of V_tensor - flattened_value,
        # but policy loss should not optimize value parameters.
        #
        # NOTE: DO NOT use this approach, value_loss depends on criticism
        # criticism = self.TD_tensor
        #
        # Alternatively, call tf.stop_gradient().
        criticism = self.V_tensor - flattened_value
        assert log_policy.shape.as_list() == criticism.shape.as_list(), "shape match failure: log(Pi) {} criticism {}".format(log_policy.shape, criticism.shape)
        if 'no_critic' in advcore.args.train:
            policy_per_sample = log_policy
        elif 'abs_critic' in advcore.args.train:
            policy_per_sample = log_policy * tf.abs(tf.stop_gradient(criticism))
        else:
            policy_per_sample = log_policy * tf.stop_gradient(criticism)
        policy_loss = tf.reduce_sum(-policy_per_sample)
        policy_loss = policy_loss * 10.0 # HACKING: Larger weight
        # policy_loss = -policy_loss # A3C paper uses gradient ascend, which means we need to minimize the NEGATIVE of the original
        # Value loss
        value_loss = tf.nn.l2_loss(criticism)

        self.print("V_tensor {} AdvCore.value {}".format(self.V_tensor.shape, flattened_value.shape))
        tf.summary.scalar('value_loss', value_loss)
        self.loss = value_loss
        if 'qonly' not in advcore.args.train:
            tf.summary.scalar('policy_loss', policy_loss)
            self.loss += policy_loss
        self._raw_policy = advcore.policy
        self._policy = policy
        self._criticism = criticism
        self._log_policy = log_policy
        self._policy_per_sample = policy_per_sample
        '''
        Debug: minimize w.r.t. value loss
        '''
        # self.loss = value_loss
        return self.loss

    '''
    New training function.
    '''
    def train(self, envir, sess, tid=None, tmax=-1):
        rlsamples, final_state = self.sample_minibatch(envir, sess, tid, tmax)
        self.a2c(envir=envir,
                 sess=sess,
                 vstates=[s.vstate for s in rlsamples] + [final_state.vstate],
                 action_indices=[s.action_index for s in rlsamples],
                 ratios=[s.ratio for s in rlsamples],
                 rewards=[s.combined_reward for s in rlsamples],
                 values=[s.value for s in rlsamples],
                 reaching_terminal=rlsamples[-1].reaching_terminal
                )

    '''
    Private function that performs the training
    '''
    def a2c(self, envir, sess, vstates, action_indices, ratios, rewards, values, reaching_terminal, pprefix=""):
        if values is None:
            # Need to Re-evaluate it
            values = self.advcore.evaluate(vstates[:-1], sess, self.flattened_value)
        assert len(vstates) == len(action_indices) + 1, "[a2c] SAS sequence was not satisfied, len(vstates) = {}, len(action_indices) = {}".format(len(vstates), len(action_indices))
        assert len(vstates) == len(values) + 1, "[a2c] len(S) != len(V) + 1"
        assert len(action_indices) == len(rewards), "[a2c] len(A) != len(rewards) "
        advcore = self.advcore
        V = 0.0
        if not reaching_terminal:
            V = np.asscalar(advcore.evaluate([vstates[-1]], sess, tensors=[advcore.value])[0])
            self.print('> V from advcore.evaluate {}'.format(V))
        # V = values[-1] # Guaranteed by A2CSampler.sample_minibatch

        # action_indices.reverse()
        # rewards.reverse()
        # values.reverse()
        r_action_indices = action_indices[::-1]
        r_rewards = rewards[::-1]
        r_values = values[::-1]

        batch_adist = []
        batch_td = []
        batch_V = []
        '''
        Calculate the per-step "true" value for current iteration
        '''
        '''
        print('[{}] R start with {}'.format(self.worker_thread_index, R))
        '''
        for (ai, ri, Vi) in zip(r_action_indices, r_rewards, r_values):
            V = ri + self.gamma * V if self.gamma > 0 else ri + V + self.gamma
            td = V - Vi
            self.print("{}V(env+ar) {} V(nn) {} reward {}".format(pprefix, V, Vi, ri))
            batch_td.append(td)
            batch_V.append(V)
        batch_rgb = [state[0] for state in vstates]
        batch_dep = [state[1] for state in vstates]
        if self.verbose_training:
            print("values {}".format(values))
            print("rewards {}".format(rewards))
            print("action_indices {}".format(action_indices))
            print("batch_rgb {}".format(batch_rgb))
            print('{}batch_a[0] {}'.format(pprefix, batch_adist[0]))
            print('{}batch_V {}'.format(pprefix, batch_V))
        '''
        Always reverse, the RLEnv need this sequential info for training.
        '''
        batch_td.reverse()
        batch_V.reverse()
        batch_adist = rlutil.actions_to_adist_array(action_indices, dim=self.action_space_dimension)
        dic = {
                advcore.rgb_1: batch_rgb[:-1],
                advcore.dep_1: batch_dep[:-1],
                advcore.rgb_2: batch_rgb[1:],
                advcore.dep_2: batch_dep[1:],
                advcore.action_tensor : batch_adist,
                self.TD_tensor: batch_td,
                self.V_tensor: batch_V
              }
        if self.batch_normalization is not None:
            dic[self.batch_normalization] = True
        if advcore.ratios_tensor is not None:
            dic[advcore.ratios_tensor] = ratios
        if advcore.using_lstm:
            dic.update({
                advcore.lstm_states_in.c : advcore.current_lstm.c,
                advcore.lstm_states_in.h : advcore.current_lstm.h,
                advcore.lstm_len : len(batch_rgb[:-1])
                       })
        self.print('{}batch_td {}'.format(pprefix, batch_td))
        self.print('{}batch_V {}'.format(pprefix, batch_V))
        self.print('{}action_indices {}'.format(pprefix, action_indices))
        self.dispatch_training(sess, dic)
        # advcore.train(sess, batch_rgb, batch_dep, batch_adist)
        # FIXME: Re-enable summary after joint the two losses.
        '''
        summary = sess.run(self.summary_op)
        self.train_writer.add_summary(summary, self.global_step)
        '''
        c,l,bp,p,raw,smraw = curiosity.sess_no_hook(sess, [self._criticism, self._log_policy, self._policy_per_sample, self._policy, self._raw_policy, advcore.softmax_policy], feed_dict=dic)
        print("policy_output_raw {}".format(raw))
        print("policy_output_smraw {}".format(smraw))
        print("policy_output_flatten {}".format(p))
        print("criticism {}".format(c))
        print("log_policy {}".format(l))
        print("policy_per_sample {}".format(bp))
        return batch_V

    def dispatch_training(self, sess, dic):
        if self.summary_op is not None:
            _, summary, gs = sess.run([self.train_op, self.summary_op, self.global_step], feed_dict=dic)
            self.train_writer.add_summary(summary, gs)
        else:
            sess.run(self.train_op, feed_dict=dic)

    def train_by_samples(self, envir, sess, states, action_indices, ratios, trewards, reaching_terminal, pprefix):
        advcore = self.advcore
        trimmed_states = states[:-1]
        if len(trimmed_states) <= 0:
            return
        arewards = advcore.get_artificial_from_experience(sess, states, action_indices, ratios, pprefix)
        arewards = np.reshape(arewards, newshape=(-1)).tolist()
        [values] = advcore.evaluate(trimmed_states, sess, [advcore.value])
        values = np.reshape(values, newshape=(-1)).tolist()
        '''
        self.print(pprefix, '> ARewards {}'.format(arewards))
        # print(pprefix, '> Values {}'.format(values))
        self.print(pprefix, '> Values list {}'.format(values))
        '''
        rewards = []
        assert len(trewards) == len(arewards), "Artificial rewards' size should match true rewards'"
        for (tr,ar) in zip(trewards, arewards):
            rewards.append(tr+ar)
        self.print(pprefix, '> Rewards {}'.format(rewards))
        bv = self.a2c(envir, sess, states, action_indices, ratios, rewards, values, reaching_terminal, pprefix)
        '''
        [valuesafter] = advcore.evaluate(trimmed_states, sess, [advcore.value])
        valuesafter = np.reshape(valuesafter, newshape=(-1)).tolist()
        self.print(pprefix, '> [DEBUG] Values before training {}'.format(values))
        self.print(pprefix, '> [DEBUG] Values target {}'.format(bv))
        self.print(pprefix, '> [DEBUG] Values after training {}'.format(valuesafter))
        '''

    '''
    a2c_erep: A2C Training with Expreience REPlay
    '''
    def a2c_erep(self, envir, sess, pprefix):
        states, action_indices, ratios, trewards, reaching_terminal = envir.sample_in_erep(pprefix)
        if len(action_indices) == 0:
            return
        self.train_by_samples(envir=envir,
                sess=sess,
                states=states,
                action_indices=action_indices,
                ratios=ratios,
                trewards=trewards,
                reaching_terminal=reaching_terminal,
                pprefix=pprefix)

QUEUE_CAPACITY = 16

'''
A2C Trainer with Dedicated Training Thread
NOTE: NOT VERY EFFECTIVE BECAUSE OF GIL
'''
class A2CTrainerDTT(A2CTrainer):
    class Arguments:
        def __init__(self, dic=None, sess=None):
            self.dic = dic
            self.sess = sess

    def __init__(self,
            advcore,
            tmax,
            gamma,
            learning_rate,
            ckpt_dir,
            global_step=None,
            entropy_beta=0.01,
            debug=True,
            batch_normalization=None,
            period=1,
            total_number_of_replicas=None,
            LAMBDA=0.5,
            train_everything=False
            ):
        super(A2CTrainerDTT, self).__init__(
                advcore=advcore,
                tmax=tmax,
                gamma=gamma,
                learning_rate=learning_rate,
                ckpt_dir=ckpt_dir,
                global_step=global_step,
                entropy_beta=entropy_beta,
                debug=debug,
                batch_normalization=batch_normalization,
                period=period,
                total_number_of_replicas=total_number_of_replicas,
                LAMBDA=LAMBDA,
                train_everything=train_everything)
        self.Q = queue.Queue(QUEUE_CAPACITY)
        self.dtt = threading.Thread(target=self.dedicated_training)
        self.dtt.start()

    def __del__(self):
        self.Q.put(self.Arguments())
        print("[A2CTrainerDTT] Waiting for DTT")
        self.dtt.join()
        print("[A2CTrainerDTT] DTT waited")

    '''
    override the synchronous version defined in A2CTrainer
    '''
    def dispatch_training(self, sess, dic):
        self.Q.put(self.Arguments(dic, sess))

    def dedicated_training(self):
        while True:
            a = self.Q.get()
            if a.dic is None:
                break
            super(A2CTrainerDTT, self).dispatch_training(a.sess, a.dic)

