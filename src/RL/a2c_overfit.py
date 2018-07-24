from __future__ import print_function
from a2c import A2CTrainer
import tensorflow as tf
import rlenv
import numpy as np
import rlutil

class OverfitTrainer(A2CTrainer):
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
        super(OverfitTrainer, self).__init__(
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
        self.all_samples_cache = []
        self.minibatch_index = 0

    def train(self, envir, sess, tid=None, tmax=-1):
        if not self.all_samples_cache:
            print("Caching samples")
            while True:
                rlsamples, final_state = self.sample_minibatch(envir, sess, tid, tmax)
                self.all_samples_cache.append((rlsamples, final_state))
                if rlsamples[-1].reaching_terminal:
                    break
        rlsamples, final_state = self.all_samples_cache[self.minibatch_index % len(self.all_samples_cache)]
        self.minibatch_index += 1
        self.a2c(envir=envir,
                 sess=sess,
                 vstates=[s.vstate for s in rlsamples] + [final_state.vstate],
                 action_indices=[s.action_index for s in rlsamples],
                 ratios=[s.ratio for s in rlsamples],
                 rewards=[s.combined_reward for s in rlsamples],
                 values=[s.value for s in rlsamples],
                 reaching_terminal=rlsamples[-1].reaching_terminal
                )
