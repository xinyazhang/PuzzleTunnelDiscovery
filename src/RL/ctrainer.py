import tensorflow as tf
import rlenv
import rlutil
import numpy as np
import uw_random
import pyosr

'''
NOTE: CLARIFY ABOUT --samplebatching AND --batch
      In pretrain-d.py, --batch defines how many frames to generate for each
      sampled initial state, and hence decide how many actions to sample for
      each initial state.

      On the other hand --samplebatching defines how many (frame, next frame)
      samples to aggregrate into one training operation.

      THIS HAS CHANGED IN curiosity-rl, we always use --batch, and it has
      different semantics under different trainers.

      For CTrainer, we always assume one action per initial state. and --batch
      determines how many samples to aggregrate into one training action.
'''

class CTrainer(object):
    def __init__(self,
            advcore,
            batch,
            learning_rate,
            ckpt_dir,
            period,
            global_step
            ):
        self.advcore = advcore
        self.batch = batch
        self.period = period
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.loss = self.build_loss(advcore)
        self.global_step = global_step
        tf.summary.scalar('curiosity_loss', self.loss)
        if ckpt_dir is not None:
            self.summary_op = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(ckpt_dir + '/summary', tf.get_default_graph())
        var_list = self._get_variables_to_train(advcore)
        print('Variables to Train {}'.format(var_list))
        self.train_op = self.optimizer.minimize(self.loss, global_step=global_step, var_list=var_list)
        self.action_space_dimension = uw_random.DISCRETE_ACTION_NUMBER
        self.actionset = None
        self.sample = None
        self.sample_limit = -1

    def _get_variables_to_train(self, advcore):
        return advcore.curiosity_params

    def build_loss(self, advcore):
        return tf.reduce_sum(advcore.curiosity)

    def render(self, envir, state):
        envir.qstate = state
        return envir.vstate

    def set_action_set(self, actionset):
        self.actionset = actionset

    def attach_samplein(self, samplein):
        self.sample = np.load(samplein)
        self.sample_iter = 0

    def limit_samples_to_use(self, limit):
        self.sample_limit = limit

    def _sample_state(self, envir):
        if self.sample is None:
            return [uw_random.gen_unit_init_state(envir.r, scale=0.5) for i in range(self.batch)]
        sampleset = self.sample['Q']
        if self.sample_limit > 0:
            sampleset = sampleset[:self.sample_limit]
        ret = np.take(sampleset,
                indices=range(self.sample_iter, self.sample_iter + self.batch),
                axis=0, mode='wrap')
        self.sample_iter += self.batch
        return ret

    def _sample_actions(self):
        if self.actionset is None:
            return np.random.randint(0, high=self.action_space_dimension, size=(self.batch))
        return np.random.choice(self.actionset, size=(self.batch))

    def _extra_tensor(self):
        return []

    def _process_extra(self, dic, rtups):
        pass

    def train(self, envir, sess, tid=None):
        advcore = self.advcore
        pprefix = "[{}] ".format(tid) if tid is not None else ""

        states = self._sample_state(envir)
        actions = self._sample_actions()
        ntuples = [envir.peek_act(actions[i], pprefix, states[i]) for i in range(self.batch)]
        images = [self.render(envir, state) for state in states]
        nimages = [self.render(envir, ntuples[i][0]) for i in range(self.batch)]
        batch_rgb_1 = [image[0] for image in images]
        batch_dep_1 = [image[1] for image in images]
        batch_rgb_2 = [image[0] for image in nimages]
        batch_dep_2 = [image[1] for image in nimages]
        adists_array = rlutil.actions_to_adist_array(actions)
        dic = {
                advcore.action_tensor : adists_array,
                advcore.rgb_1 : batch_rgb_1,
                advcore.dep_1 : batch_dep_1,
                advcore.rgb_2 : batch_rgb_2,
                advcore.dep_2 : batch_dep_2,
              }
        # print(values)
        # print(values.shape)
        to_eva = [self.loss, self.train_op, self.summary_op, self.global_step] + self._extra_tensor()
        rtups = sess.run(to_eva, feed_dict=dic)
        loss = rtups[0]
        summary = rtups[2]
        step = rtups[3]
        self.train_writer.add_summary(summary, step)
        self._process_extra(dic, rtups)
        print("Current loss {}".format(loss))
        if envir.steps_since_reset >= self.period * self.batch:
            envir.reset()
