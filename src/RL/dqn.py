# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
from __future__ import print_function
import tensorflow as tf
import rlenv
import rlutil
import util
import numpy as np
import multiprocessing as mp
import cPickle as pickle
import time
import curiosity

class DQNTrainer(object):

    def __init__(self,
                 advcore,
                 args,
                 learning_rate,
                 batch_normalization=None):
        self.advcore = advcore
        self.action_space_dimension = int(advcore.policy.shape[-1])
        advcore.softmax_policy # create normalized policy tensor
        self.args = args
        if args.train == 'dqn_overfit':
            self.sampler = rlenv.CachedMiniBatchSampler(advcore=advcore, args=args)
        elif 'whole_traj' in args.debug_flags:
            self.sampler = rlenv.WholeTrajSampler(advcore=advcore)
        else:
            self.sampler = rlenv.MiniBatchSampler(advcore=advcore, tmax=args.batch)
        self.batch_normalization = batch_normalization
        global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step
        '''
        Design idea: reuse curiosity.CuriosityRL to implement DQN algorithm
        Key choice: use CuriosityRL.policy as the Q function approximator
        '''
        self.dqn_out = advcore.polout
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if args.period > 1:
            self.optimizer = tf.train.SyncReplicasOptimizer(self.optimizer,
                    replicas_to_aggregate=args.period,
                    total_num_replicas=args.localcluster_nsampler)
        LAMBDA_1 = 1
        self.loss = LAMBDA_1 * self._build_loss(advcore)
        if args.train != 'dqn_overfit':
            LAMBDA_2 = 1
            self.loss += LAMBDA_2 * advcore.build_loss()

        '''
        Train everything by default
        '''
        if batch_normalization is not None:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
        else:
            self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
        grad_op = self.optimizer.compute_gradients(self.loss)
        '''
        self._debug_grad_op = []
        for tup in grad_op:
            if tup[0] is not None:
                self._debug_grad_op.append(tup)
        print("self._debug_grad_op {}".format(self._debug_grad_op))
        '''
        ckpt_dir = args.ckptdir
        if ckpt_dir is not None:
            self.summary_op = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(ckpt_dir + '/summary', tf.get_default_graph())
        else:
            self.summary_op = None
            self.train_writer = None

    def _log(self, text):
        print(text)

    @property
    def total_iter(self):
        return self.args.iter

    def _build_loss(self, advcore):
        self.Adist_tensor = advcore.action_tensor
        self.Q_tensor = tf.placeholder(tf.float32, shape=[advcore.batch_size, 1, self.action_space_dimension], name='QPh')
        # Q_output = tf.reduce_sum(tf.multiply(self.Adist_tensor, self.dqn_out), axis=[1,2])
        dqn_loss = tf.nn.l2_loss(self.Q_tensor - self.dqn_out)
        tf.summary.scalar('dqn_loss', dqn_loss)

        self._debug_fvmag = tf.reduce_sum(advcore.model.cur_mvfeatvec, axis=[1,2])
        return dqn_loss

    def train(self, envir, sess, tid=None, tmax=-1):
        rlsamples, final_state = self.sampler.sample_minibatch(envir, sess, tid, tmax)
        action_indices = [s.action_index for s in rlsamples]
        batch_adist = rlutil.actions_to_adist_array(action_indices, dim=self.action_space_dimension)

        advcore = self.advcore
        # Bootstrap V
        if final_state.is_terminal:
            V = 0.0
        else:
            [allq] = advcore.evaluate([final_state.vstate], sess, tensors=[advcore.policy])
            assert allq.shape == (1, 1, 12), "advcore.policy output shape is not (1,1,12) but {}".format(allq.shape)
            allq = allq[0][0][self.args.actionset] # Eliminate unselected actions
            V = np.amax(allq)
            # self.print('> V bootstraped from advcore.evaluate {}'.format(V))
        # Calculate V from sampled Traj.
        rewards = [s.combined_reward for s in rlsamples]
        r_rewards = rewards[::-1]
        batch_V = []
        GAMMA = self.args.GAMMA
        for r in r_rewards:
            V = r + GAMMA * V if GAMMA > 0 else r + V + GAMMA
            batch_V.append(V)
        batch_V.reverse()
        batch_Q = rlutil.actions_to_adist_array(action_indices, dim=self.action_space_dimension, hotvector=batch_V)
        # Prepare Input for Vision
        batch_rgb = [s.vstate[0] for s in rlsamples]
        batch_rgb.append(final_state.vstate[0])
        batch_dep = [s.vstate[1] for s in rlsamples]
        batch_dep.append(final_state.vstate[1])

        ndic = {
                'rgb' : batch_rgb,
                'dep' : batch_dep,
                'adist' : batch_adist,
                'aindex' : action_indices,
                'q' : np.array([s.qstate for s in rlsamples] + [final_state.qstate]),
                'V' : batch_V,
                'Q' : batch_Q,
                'rewards' : rewards,
               }
        self.dispatch_training(sess, ndic)

    def dispatch_training(self, sess, ndic, debug_output=True):
        advcore = self.advcore
        dic = {
                advcore.rgb_1: ndic['rgb'][:-1],
                advcore.dep_1: ndic['dep'][:-1],
                advcore.rgb_2: ndic['rgb'][1:],
                advcore.dep_2: ndic['dep'][1:],
                advcore.action_tensor : ndic['adist'],
                self.Q_tensor: ndic['Q']
              }
        if debug_output:
            [raw] = curiosity.sess_no_hook(sess, [advcore.policy], feed_dict=dic)
            print("action input {}".format(ndic['aindex']))
            print("action distribution {}".format(ndic['adist']))
            print("reward output {}".format(ndic['rewards']))
            print("V {}".format(ndic['V']))
            print("policy_output_raw {}".format(raw))
        if self.summary_op is not None:
            self._log("running training op")
            _, summary, gs, fvmag = sess.run([self.train_op, self.summary_op, self.global_step, self._debug_fvmag], feed_dict=dic)
            self._log("training op for gs {} finished".format(gs))
            print("[DEBUG] fv_mag: {}".format(fvmag))
            # grads = curiosity.sess_no_hook(sess, self._debug_grad_op, feed_dict=dic)
            # print("[DEBUG] grads: {}".format(grads))
            self.train_writer.add_summary(summary, gs)
        else:
            sess.run(self.train_op, feed_dict=dic)

class DQNTrainerMP(DQNTrainer):

    def __init__(self,
                 advcore,
                 args,
                 learning_rate,
                 batch_normalization=None):
        super(DQNTrainerMP, self).__init__(
                advcore=advcore,
                args=args,
                learning_rate=learning_rate,
                batch_normalization=batch_normalization)
        self._MPQ = None
        self._task_index = None

    def install_mpqueue_as(self, mpqueue, task_index):
        self._MPQ = mpqueue
        self._task_index = task_index
        self._logfile = open(self.args.ckptdir + '{}.out'.format(task_index), 'w')
        self._sample_dumpdir = self.args.ckptdir + "sample-{}/".format(task_index)
        util.mkdir_p(self._sample_dumpdir)
        self._sample_index = 0

    def _log(self, text):
        print("[{}] {}".format(time.time(), text), file=self._logfile)
        self._logfile.flush()

    def _dump_sample(self, ndic):
        fn = "{}/{}".format(self._sample_dumpdir, self._sample_index)
        np.savez(fn, Qs=ndic['q'], As=ndic['aindex'], Vs=ndic['V'])
        self._sample_index += 1

    '''
    TODO and FIXME: CODE REUSE
    The following part is identicial to MPA2CTrainer
    '''

    @property
    def is_chief(self):
        assert self._task_index is not None, "is_chief is called before install_mpqueue_as"
        return self._task_index == 0

    @property
    def total_iter(self):
        assert self._task_index is not None, "total_iter is called before install_mpqueue_as"
        if self.is_chief:
            return self.args.iter * self.args.localcluster_nsampler
        return self.args.iter

    def train(self, envir, sess, tid=None, tmax=-1):
        if self.is_chief:
            # Collect generated sample
            self._log("waiting for training data")
            pk = self._MPQ.get()
            self._log("packet received")
            ndic = pickle.loads(pk)
            self._log("training data unpacked, batch size {}".format(len(ndic['V'])))
            super(DQNTrainerMP, self).dispatch_training(sess, ndic, debug_output=False)
            self._log("minibatch trained, batch size {}".format(len(ndic['V'])))
        else:
            # Generate samples
            self._log("Generating minibatch")
            super(DQNTrainerMP, self).train(envir, sess, tid, tmax)

    def dispatch_training(self, sess, ndic):
        assert not self.is_chief, "DQNTrainerMP.dispatch_training must not be called by chief"
        pk = pickle.dumps(ndic, protocol=-1)
        self._MPQ.put(pk)
        self._log("Minibatch put into queue")
        self._dump_sample(ndic)
