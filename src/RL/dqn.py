from __future__ import print_function
import tensorflow as tf
import rlenv
import numpy as np
import multiprocessing as mp
import cPickle as pickle

class DQNTrainer(object):

    def __init__(self,
                 advcore,
                 args,
                 learning_rate,
                 batch_normalization=None):
        self.advcore = advcore
        self.args = args
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
                    replicas_to_aggregate=period,
                    total_num_replicas=total_number_of_replicas)
        LAMBDA_1 = 1
        LAMBDA_2 = 1
        self.loss = LAMBDA_1 * self._build_loss(advcore) + LAMBDA_2 * advcore.build_loss()
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

    @property
    def total_iter(self):
        return self.args.iter

    def _build_loss(self, advcore):
        self.Adist_tensor = advcore.action_tensor
        self.Q_tensor = tf.placeholder(tf.float32, shape=[None], name='VPh')
        Q_output = tf.reduce_sum(tf.multiply(self.Adist_tensor, self.dqn_out), axis=[1,2])
        dqn_loss = tf.nn.l2_loss(self.Q_tensor - Q_output)
        tf.summary.scalar('dqn_loss', dqn_loss)
        return dqn_loss

    def train(self, envir, sess, tid=None, tmax=-1):
        rlsamples, final_state = self.sampler.sample_minibatch(envir, sess, tid, tmax)
        batch_adist = rlutil.actions_to_adist_array(action_indices, dim=self.action_space_dimension)

        advcore = self.advcore
        # Bootstrap V
        if final_state.is_terminal:
            V = 0.0
        else:
            V = np.asscalar(advcore.evaluate([final_state.vstate], sess, tensors=[advcore.value])[0])
            self.print('> V bootstraped from advcore.evaluate {}'.format(V))
        # Calculate V from sampled Traj.
        r_rewards = [s.combined_reward for s in rlsamples][::-1]
        batch_V = []
        GAMMA = self.args.GAMMA
        for r in r_rewards:
            V = r + GAMMA * V if GAMMA > 0 else r + V + GAMMA
            batch_V.append(V)
        batch_V.reverse()
        # Prepare Input for Vision
        batch_rgb = [s.vstate[0] for s in rlsamples]
        batch_dep = [s.vstate[1] for s in rlsamples]

        ndic = {
                'rgb' : batch_rgb,
                'dep' : batch_dep,
                'adist' : batch_adist,
                'V' : batch_V,
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
                self.V_tensor: ndic['V']
              }
        if self.summary_op is not None:
            _, summary, gs = sess.run([self.train_op, self.summary_op, self.global_step], feed_dict=dic)
            self.train_writer.add_summary(summary, gs)
        else:
            sess.run(self.train_op, feed_dict=dic)

class DQNTrainerMP(object):

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
            pk = self._MPQ.get()
            ndic = pickle.loads(pk)
            super(DQNTrainerMP, self).dispatch_training(sess, ndic, debug_output=False)
        else:
            # Generate samples
            super(DQNTrainerMP, self).train(envir, sess, tid, tmax)

    def dispatch_training(self, sess, ndic):
        assert not self.is_chief, "DQNTrainerMP.dispatch_training must not be called by chief"
        pk = pickle.dumps(ndic, protocol=-1)
        self._MPQ.put(pk)
