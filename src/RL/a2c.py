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

class A2CTrainer(object):
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
                 LAMBDA=0.5,
                 train_everything=False
                ):
        self.advcore = advcore
        self.a2c_tmax = tmax
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
            self.optimizer = tf.train.SyncReplicasOptimizer(self.optimizer, replicas_to_aggregate=period)
        LAMBDA = 0.5
        self.loss = LAMBDA * self.build_loss(advcore)
        print("self.loss 1 {}".format(self.loss))
        tf.summary.scalar('a2c_loss', self.loss)
        self.loss += (1 - LAMBDA) * advcore.build_loss()
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
        assert ckpt_dir is not None, "A2CTrainer: ckpt_dir must not be None"
        if ckpt_dir is not None:
            self.summary_op = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(ckpt_dir + '/summary', tf.get_default_graph())
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
        self.print('rindices {}'.format(rindices))
        action_entropy = tf.reduce_sum(tf.multiply(log_policy, self.Adist_tensor),
                reduction_indices=rindices)
        entropy = -tf.reduce_sum(policy * log_policy, reduction_indices=rindices)
        self.print('action_entropy shape {}'.format(action_entropy.shape))

        # Why do we add entropy to our loss?
        # policy_loss_per_step = tf.reduce_sum(action_entropy * self.TD_tensor) + entropy * self.entropy_beta
        policy_loss_per_step = tf.reduce_sum(action_entropy * self.TD_tensor)
        policy_loss = -tf.reduce_sum(policy_loss_per_step)
        '''

        '''
        New policy loss
        '''
        # Need V as critic
        # advcore.value's shape is (B,V,1)
        flattened_value = tf.reshape(advcore.value, [-1])

        # Pick out the sampled action from policy output
        # Shape: (B,V,A)
        policy = tf.multiply(advcore.softmax_policy, self.Adist_tensor)
        policy = tf.reduce_sum(policy, axis=[1,2]) # Shape: (B) afterwards
        log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
        criticism = self.V_tensor - flattened_value
        policy_loss = tf.reduce_sum(log_policy * criticism)
        policy_loss = -policy_loss # A3C paper uses gradient ascend, which means we need to minimize the NEGATIVE of the original

        # Value loss
        value_loss = tf.nn.l2_loss(criticism)
        self.print("V_tensor {} AdvCore.value {}".format(self.V_tensor.shape, flattened_value.shape))
        self.loss = policy_loss+value_loss
        '''
        Debug: minimize w.r.t. value loss
        '''
        # self.loss = value_loss
        return self.loss

    '''
    Train the network

    This method interacts with RLEnv object to collect truths
    '''
    def train(self, envir, sess, tid=None, tmax=-1):
        if tmax < 0:
            tmax = self.a2c_tmax
        if tid is None:
            pprefix = ""
        else:
            pprefix = "[{}] ".format(tid)
        advcore = self.advcore
        reaching_terminal = False
        states = []
        action_indices = []
        ratios = []
        actual_rewards = []
        combined_rewards = []
        values = []
        lstm_begin = advcore.get_lstm()
        for i in range(tmax):
            policy, value = advcore.evaluate([envir.vstate], sess, [advcore.softmax_policy, advcore.value])
            '''
            Pick up the only frame
            '''
            self.print('{}unmasked pol {} shape {}; val {} shape {}'.format(pprefix, policy, policy.shape, value, value.shape))
            # self.print('{}masked pol {} shape {};'.format(pprefix, policy, policy.shape))
            policy = policy[0][0] # Policy View from first qstate and first view
            value = np.asscalar(value) # value[0][0][0]
            lstm_next = advcore.get_lstm()
            action_index = advcore.make_decision(envir, policy, pprefix)
            states.append(envir.vstate)
            '''
            FIXME: Wait, shouldn't be policy?
            '''
            action_indices.append(action_index)
            values.append(value)

            self.print("{}Peeking action".format(pprefix))
            nstate,reward,reaching_terminal,ratio = envir.peek_act(action_index, pprefix=pprefix)
            if reward < 0:
                # print("vstate shape {}".format(envir.vstate.shape))
                imsave('coldu/collison_dump_{}.png'.format(self.dbg_sample_peek), envir.vstate[0][0])
                np.savez('coldu/collison_dump_{}.npz'.format(self.dbg_sample_peek),
                        Q1=envir.qstate,
                        Q2=nstate,
                        P=envir.get_perturbation(),
                        A=action_index)
                self.dbg_sample_peek += 1
            ratios.append(ratio)
            actual_rewards.append(reward)
            # print("action peeked {} ratio {} terminal? {}".format(nstate, ratio, reaching_terminal))
            reward += advcore.get_artificial_reward(envir, sess,
                    envir.qstate, action_index, nstate, ratio, pprefix)
            combined_rewards.append(reward)
            '''
            Store Exprience
            '''
            envir.store_erep(states[-1], envir.qstate, action_indices[-1], ratios[-1],
                             actual_rewards[-1],
                             reaching_terminal,
                             envir.get_perturbation()
                             )
            '''
            Experience Replay
            '''
            self.a2c_erep(envir, sess, pprefix)
            if reaching_terminal:
                break
            '''
            Leave for training because of collision
            if ratio == 0:
                break
            '''
            advcore.set_lstm(lstm_next) # AdvCore next frame
            envir.qstate = nstate # Envir Next frame
        advcore.set_lstm(lstm_begin)
        # self.a2c(envir, sess, action_indices, states, combined_rewards, values, reaching_terminal, pprefix)
        # print("> states length {}, shape {}".format(len(states), states[0][0].shape))
        # print("> action_indices length {}, tmax {}".format(len(action_indices), tmax))
        states.append(envir.vstate)
        self.train_by_samples(envir=envir,
                sess=sess,
                states=states,
                action_indices=action_indices,
                ratios=ratios,
                trewards=actual_rewards,
                reaching_terminal=reaching_terminal,
                pprefix=pprefix)
        advcore.set_lstm(lstm_next)

        if reaching_terminal:
            '''
            Add the final state to erep cache
            '''
            envir.store_erep(states[-1], envir.qstate,
                    -1, -1, -1,
                    reaching_terminal,
                    envir.get_perturbation()
                    )
            '''
            Train the experience in sample_cap iterations
            '''
            for i in range(envir.erep_sample_cap):
                self.a2c_erep(envir, sess, pprefix)
            envir.reset()
            assert len(envir.erep_action_indices) == 0, "Exp Rep is not cleared after reaching terminal"

    '''
    Private function that performs the training
    '''
    def a2c(self, envir, sess, vstates, action_indices, ratios, rewards, values, reaching_terminal, pprefix=""):
        advcore = self.advcore
        V = 0.0
        if not reaching_terminal:
            V = np.asscalar(advcore.evaluate([envir.vstate], sess, tensors=[advcore.value])[0])
            self.print('> V from advcore.evaluate {}'.format(V))

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
            V = ri + self.gamma * V
            td = V - Vi
            self.print("{}V(env+ar) {} V(nn) {} reward {}".format(pprefix, V, Vi, ri))
            adist = np.zeros(shape=(1, self.action_space_dimension),
                    dtype=np.float32)
            adist[0, ai] = 1.0

            batch_adist.append(adist)
            batch_td.append(td)
            batch_V.append(V)
        batch_rgb = [state[0] for state in vstates]
        batch_dep = [state[1] for state in vstates]
        if self.verbose_training:
            self.print('{}batch_a[0] {}'.format(pprefix, batch_adist[0]))
            self.print('{}batch_V {}'.format(pprefix, batch_R))
        '''
        Always reverse, the RLEnv need this sequential info for training.
        '''
        batch_adist.reverse()
        batch_td.reverse()
        batch_V.reverse()
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
        self.dispatch_training(sess, dic)
        # advcore.train(sess, batch_rgb, batch_dep, batch_adist)
        # FIXME: Re-enable summary after joint the two losses.
        '''
        summary = sess.run(self.summary_op)
        self.train_writer.add_summary(summary, self.global_step)
        '''
        return batch_V

    def dispatch_training(self, sess, dic):
        _, summary, gs = sess.run([self.train_op, self.summary_op, self.global_step], feed_dict=dic)
        self.train_writer.add_summary(summary, gs)

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
