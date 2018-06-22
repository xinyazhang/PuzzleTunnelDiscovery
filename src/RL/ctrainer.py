import tensorflow as tf
import rlenv
import rlutil
import numpy as np
import uw_random
import pyosr

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
        self.train_op = self.optimizer.minimize(self.loss, global_step=global_step, var_list=advcore.curiosity_params)
        self.action_space_dimension = uw_random.DISCRETE_ACTION_NUMBER
        self.actionset = None
        self.sample = None
        self.sample_limit = -1

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
        loss, _, summary, step = sess.run([self.loss, self.train_op, self.summary_op, self.global_step], feed_dict=dic)
        self.train_writer.add_summary(summary, step)
        print("Current loss {}".format(loss))
        if envir.steps_since_reset >= self.period * self.batch:
            envir.reset()
