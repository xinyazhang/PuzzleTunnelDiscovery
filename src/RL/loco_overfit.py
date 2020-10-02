# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
from __future__ import print_function
import tensorflow as tf
import rlenv
import rlutil
import util
import numpy as np
import multiprocessing as mp
import cPickle as pickle
import time
import rlcaction
import os

class LocoOverfitter(object):

    def __init__(self,
                 advcore,
                 args,
                 learning_rate,
                 batch_normalization=None):
        assert args.samplein, '--train loco_overfit requires --samplein'
        self.samplein_index = 0
        self.samplein_max = 0
        while os.path.exists('{}/{}.npz'.format(args.samplein, self.samplein_max)):
            self.samplein_max += 1
        print('... Found {} samples at {}'.format(self.samplein_max, args.samplein))

        self.advcore = advcore
        self.action_space_dimension = int(advcore.policy.shape[-1])
        advcore.softmax_policy # create normalized policy tensor
        self.args = args
        self.batch_normalization = batch_normalization
        global_step = tf.train.get_or_create_global_step()

        self.global_step = global_step
        self.loco_out = advcore.polout[:,:,:6]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        LAMBDA_1 = 1
        self.loss = LAMBDA_1 * self._build_loss(advcore)
        # WARNING: do NOT call advcore.build_loss()
        # Loco assumes continuous actions while advcore.build_loss assumes discrete actions

        '''
        Train everything by default
        '''
        if batch_normalization is not None:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
        else:
            self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
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
        self.locomo_tensor = tf.placeholder(tf.float32,
                shape=[advcore.batch_size, 1, 6],
                name='LocoMoPh')
        # Q_output = tf.reduce_sum(tf.multiply(self.Adist_tensor, self.dqn_out), axis=[1,2])
        loco_loss = tf.nn.l2_loss(self.locomo_tensor - self.loco_out)
        tf.summary.scalar('loco_loss', loco_loss)

        return loco_loss

    def train(self, envir, sess, tid=None, tmax=-1):
        all_vstates = []
        all_cactions = []
        args = self.args
        fn = '{}/{}.npz'.format(args.samplein, self.samplein_index)
        d = np.load(fn)
        self.samplein_index = (self.samplein_index + 1) % self.samplein_max
        for qs,ctr,caa in zip(d['QS'], d['CTR'], d['CAA']):
            assert envir.r.is_valid_state(qs)
            envir.qstate = qs
            all_vstates.append(envir.vstate)
            all_cactions.append([np.concatenate([ctr, caa], axis=-1)])
        seg = args.batch
        for i in range(0, len(all_vstates), seg):
            vstates = all_vstates[i:i+seg]
            cactions = all_cactions[i:i+seg]
            batch_rgb = [vs[0] for vs in vstates]
            batch_dep = [vs[1] for vs in vstates]
            ndic = {
                    'rgb' : batch_rgb,
                    'dep' : batch_dep,
                    'locomo' : cactions
                   }
            self.dispatch_training(sess, ndic)

    def dispatch_training(self, sess, ndic, debug_output=True):
        advcore = self.advcore
        dic = {
                advcore.rgb_1: ndic['rgb'],
                advcore.dep_1: ndic['dep'],
                self.locomo_tensor : ndic['locomo'],
              }
        if self.summary_op is not None:
            self._log("running training op")
            _, summary, gs = sess.run([self.train_op, self.summary_op, self.global_step], feed_dict=dic)
            self._log("training op for gs {} finished".format(gs))
            # grads = curiosity.sess_no_hook(sess, self._debug_grad_op, feed_dict=dic)
            # print("[DEBUG] grads: {}".format(grads))
            self.train_writer.add_summary(summary, gs)
        else:
            sess.run(self.train_op, feed_dict=dic)
