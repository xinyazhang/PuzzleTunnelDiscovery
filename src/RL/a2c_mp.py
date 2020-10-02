# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
from __future__ import print_function
from a2c import A2CTrainer
import tensorflow as tf
import rlenv
import numpy as np
import rlutil
import multiprocessing as mp
import cPickle as pickle

class MPA2CTrainer(A2CTrainer):
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
        super(MPA2CTrainer, self).__init__(
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
        self.args = advcore.args
        self._MPQ = None
        self._task_index = None

    def install_mpqueue_as(self, mpqueue, task_index):
        self._MPQ = mpqueue
        self._task_index = task_index

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
            pk = self._MPQ.get()
            ndic = pickle.loads(pk)
            super(MPA2CTrainer, self).dispatch_training(sess, ndic, debug_output=False)
        else:
            # Generate samples
            super(MPA2CTrainer, self).train(envir, sess, tid, tmax)

    def dispatch_training(self, sess, ndic):
        assert not self.is_chief, "MPA2CTrainer.dispatch_training must not be called by chief"
        pk = pickle.dumps(ndic, protocol=-1)
        self._MPQ.put(pk)
