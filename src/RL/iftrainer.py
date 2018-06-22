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

    def build_loss(self, advcore):
        BETA = 0.2
        return (1.0 - BETA) * advcore.model.get_inverse_loss(discrete=True) + BETA * tf.reduce_sum(advcore.curiosity)
