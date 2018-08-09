import tensorflow as tf
import rlenv
import numpy as np
import uw_random
import pyosr
from scipy.misc import imsave

class QTrainer:
    def __init__(self,
            advcore,
            batch,
            learning_rate,
            ckpt_dir,
            period,
            global_step,
            train_fcfe=False,
            train_everything=False
            ):
        self.gt = None
        self.advcore = advcore
        self.batch = batch
        self.period = period
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.loss = self.build_loss(advcore)
        self.global_step = global_step
        tf.summary.scalar('q_loss', self.loss)
        if ckpt_dir is not None:
            self.summary_op = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(ckpt_dir + '/summary', tf.get_default_graph())
        var_list=advcore.valparams
        if train_fcfe:
            var_list += advcore.model.cat_nn_vars['fc']
        if train_everything:
            var_list = None
        self.train_op = self.optimizer.minimize(self.loss, global_step=global_step, var_list=var_list)
        print("Training Q function over {}".format(var_list))

    @property
    def total_iter(self):
        return self.advcore.args.iter

    def build_loss(self, advcore):
        self.V_tensor = tf.placeholder(tf.float32, shape=[None], name='VPh')
        flattened_value = tf.reshape(advcore.value, [-1])
        assert "{}".format(self.V_tensor.shape) == "{}".format(flattened_value.shape), "V_tensor's shape {} does not match flattened_value's {}".format(self.V_tensor.shape, flattened_value.shape)
        return tf.nn.l2_loss(self.V_tensor - flattened_value)

    UP = np.array([0,0,1])

    def _analytic_q(self, envir, states):
        r = envir.r
        p = r.perturbation
        rot = pyosr.extract_rotation_matrix(p)
        up = rot.dot(self.UP)
        origin = p[0:3]
        # print(up)
        # print(states[0][0:3] - origin)
        return [abs(np.dot(up, state[0:3] - origin)) for state in states]

    def attach_gt(self, gt):
        # GT format should match the output from rl-precalcmap.py
        self.gt_file = np.load(gt)
        self.gt = {}
        self.gt['V'] = np.copy(self.gt_file['V'])
        self.gt['D'] = np.copy(self.gt_file['D'])
        self.gt_iter = 0

    def sample(self, envir):
        if self.gt is None:
            assert self.advcore.args.train != 'q_overfit', 'q_overfit expects --samplein'
            states = [uw_random.gen_unit_init_state(envir.r) for i in range(self.batch)]
            values = self._analytic_q(envir, states)
            values = np.array(values, dtype=np.float)
            return states, values
        states = np.take(self.gt['V'],
                indices=range(self.gt_iter, self.gt_iter + self.batch),
                axis=0, mode='wrap')
        values = np.take(self.gt['D'],
                indices=range(self.gt_iter, self.gt_iter + self.batch),
                axis = 0,mode='wrap')
        self.gt_iter += self.batch
        return states, values

    def render(self, envir, state):
        envir.qstate = state
        return envir.vstate

    def train(self, envir, sess, tid=None):
        states, values = self.sample(envir)
        images = [self.render(envir, state) for state in states]
        batch_rgb = [image[0] for image in images]
        batch_dep = [image[1] for image in images]
        advcore = self.advcore
        dic = {
                advcore.rgb_1: batch_rgb,
                advcore.dep_1: batch_dep,
                self.V_tensor: values,
              }
        '''
        if self.gt_iter == self.batch: # First iteration, dump the RGB
            for i, rgb in enumerate(batch_rgb):
                imsave('qtrainer_peek_{}.png'.format(i), rgb[0])
            exit()
        '''
        # print(values)
        # print(values.shape)
        loss, _, summary, step = sess.run([self.loss, self.train_op, self.summary_op, self.global_step], feed_dict=dic)
        self.train_writer.add_summary(summary, step)
        print("Current loss {}".format(loss))
        if envir.steps_since_reset >= self.period * self.batch:
            envir.reset()
