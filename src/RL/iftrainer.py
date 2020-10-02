# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import tensorflow as tf
import rlenv
import rlutil
import numpy as np
import uw_random
import pyosr
import ctrainer

class IFTrainer(ctrainer.CTrainer):
    def __init__(self,
            advcore,
            batch,
            learning_rate,
            ckpt_dir,
            period,
            global_step
            ):
        super(IFTrainer, self).__init__(
                advcore=advcore,
                batch=batch,
                learning_rate=learning_rate,
                ckpt_dir=ckpt_dir,
                period=period,
                global_step=global_step)

    def _get_variables_to_train(self, advcore):
        return advcore.curiosity_params + advcore.model.cat_nn_vars['fc'] + advcore.model.inverse_model_params

    def build_loss(self, advcore):
        BETA = 0.02
        self.inverse_loss = advcore.model.get_inverse_loss(discrete=True)
        self.forward_loss = tf.reduce_sum(advcore.curiosity)
        self.weighted_inverse_loss  = (1.0 - BETA) * self.inverse_loss
        self.weighted_forward_loss  = BETA * self.forward_loss
        tf.summary.scalar('inverse_loss', self.inverse_loss)
        tf.summary.scalar('forward_loss', self.forward_loss)
        tf.summary.scalar('weighted_inverse_loss', self.weighted_inverse_loss)
        tf.summary.scalar('weighted_forward_loss', self.weighted_forward_loss)
        return self.weighted_inverse_loss + self.weighted_forward_loss

class ITrainer(ctrainer.CTrainer):
    def __init__(self,
            advcore,
            batch,
            learning_rate,
            ckpt_dir,
            period,
            global_step
            ):
        super(ITrainer, self).__init__(
                advcore=advcore,
                batch=batch,
                learning_rate=learning_rate,
                ckpt_dir=ckpt_dir,
                period=period,
                global_step=global_step)
        self.predicts = tf.nn.softmax(advcore.model.inverse_output_tensor)

    def _get_variables_to_train(self, advcore):
        return advcore.model.cat_nn_vars['fc'] + advcore.model.inverse_model_params

    def _extra_tensor(self):
        return [self.predicts]

    def _process_extra(self, dic, rtups):
        advcore = self.advcore
        pred = rtups[-1]
        print("ALL PRED {}".format(pred))
        pred_index = np.argmax(pred, axis=2)
        gt = dic[advcore.action_tensor]
        gt_index = np.argmax(gt, axis=2)
        assert len(pred_index) == len(gt_index), "First dimension of pred_index and gt_index should match"
        correct = 0
        total = len(gt_index)
        for i in range(total):
            correct += (1 if pred_index[i, 0] == gt_index[i, 0] else 0)
            print("Should {} Pred {}\n\tDetails:{}".format(gt_index[i, 0], pred_index[i,0], pred[i,0]))
        print("Pred Accuracy of current mini-batch {}/{} = {} % ".format(correct,
                    total, correct*100.0/total))

    def build_loss(self, advcore):
        self.inverse_loss = advcore.model.get_inverse_loss(discrete=True)
        tf.summary.scalar('inverse_loss', self.inverse_loss)
        return self.inverse_loss

