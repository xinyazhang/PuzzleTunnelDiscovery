import rlenv
import tensorflow as tf
import argparse
import config
import a2c
import numpy as np
import random

class Corridor(rlenv.IEnvironment):
    ACTION_DELTA = [-1,1]

    def __init__(self, args):
        super(Corridor, self).__init__()

        self.state = 0
        self.mag = args.res

    def qstate_setter(self, state):
        self.state = state

    def qstate_getter(self):
        return self.state

    qstate = property(qstate_getter, qstate_setter)

    @property
    def vstate(self):
        return self.state, 0

    @property
    def vstatedim(self):
        return [1]

    '''
    0 Left  (-=1)
    1 Right (+=1)
    '''
    def peek_act(self, action):
        nstate = self.state + self.ACTION_DELTA[action]
        reward = 0
        reaching_terminal = False
        if nstate > self.mag:
            reward = 1000
            reaching_terminal = True
            nstate = self.mag
        elif nstate <= -self.mag:
            reward = -1000
            nstate = -self.mag
        return nstate, reward, reaching_terminal, 1.0

    def reset(self):
        self.state = 0

class TabularRL(rlenv.IAdvantageCore):

    egreedy = 0.5
    using_lstm = False

    def __init__(self, learning_rate, args):
        super(TabularRL, self).__init__()
        self.w = args.res * 2 + 1 # -res to res

        self.action_tensor = tf.placeholder(tf.float32, shape=[None, 1, 2], name='ActionPh')
        self.rgb_1_tensor = tf.placeholder(tf.int32, shape=[None], name='Rgb1Ph')
        self.dep_1_tensor = tf.placeholder(tf.int32, shape=[None], name='Dep1Ph')
        self.rgb_2_tensor = tf.placeholder(tf.int32, shape=[None], name='Rgb2Ph')
        self.dep_2_tensor = tf.placeholder(tf.int32, shape=[None], name='Dep2Ph')

        self.polparams = tf.get_variable('polgrid', shape=[self.w, 2],
                dtype=tf.float32, initializer=tf.zeros_initializer())
        self.valparams = tf.get_variable('valgrid', shape=[self.w, 1],
                dtype=tf.float32, initializer=tf.zeros_initializer())
        self.smpolparams = tf.nn.softmax(logits=self.polparams)

        self.indices = tf.reshape(self.rgb_1_tensor, [-1]) + args.res
        self.polout = tf.gather(self.smpolparams,
            indices=self.indices)
        print('polout shape {}'.format(self.polout.shape))
        self.valout = 1000.0 * tf.gather(self.valparams,
            indices=self.indices)

        print('Polout {} Valout {}'.format(self.polout.shape, self.valout.shape))

    @property
    def softmax_policy(self):
        return tf.reshape(self.policy, [-1,1,2])

    @property
    def rgb_1(self):
        return self.rgb_1_tensor
    @property
    def rgb_2(self):
        return self.rgb_2_tensor
    @property
    def dep_1(self):
        return self.dep_1_tensor
    @property
    def dep_2(self):
        return self.dep_2_tensor
    @property
    def policy(self):
        return self.polout
    @property
    def value(self):
        return self.valout
    @property
    def policy_params(self):
        return [self.polparams]
    @property
    def value_params(self):
        return [self.valparams]
    @property
    def lstm_params(self):
        return []

    def evaluate_current(self, vstate, sess, tensors, additional_dict=None):
        rgb,dep = vstate
        dic = {
                self.rgb_1 : [rgb],
        }
        if additional_dict is not None:
            dic.update(additional_dict)
        return sess.run(tensors, feed_dict=dic)

    def make_decision(self, policy_dist):
        best = np.argmax(policy_dist, axis=-1)
        if random.random() < self.egreedy:
            ret = random.randrange(2)
        else:
            ret = best
        print('Action best {} chosen {}'.format(best, ret))
        return ret

    def get_artificial_reward(self, envir, sess, state_1, adist, state_2):
        return 0

    def train(self, sess, rgb, dep, actions):
        pass

    def lstm_next(self):
        return 0

    def set_lstm(self, lstm):
        pass

    def get_lstm(self):
        return 0

    def load_pretrain(self, sess, viewinitckpt):
        pass

def corridor_main():
    parser = argparse.ArgumentParser(description='Process some integers.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--res', help='Size of the corridor', type=int, default=8)
    parser.add_argument('--batch', metavar='NUMBER',
            help='T_MAX in A3C/A2C algo.',
            type=int, default=32)
    parser.add_argument('--iter', metavar='NUMBER',
            help='Number of iterations to train',
            type=int, default=0)
    args = parser.parse_args()

    total_epoch = args.iter

    g = tf.Graph()
    with g.as_default(), tf.device("/cpu:0"):
        global_step = tf.contrib.framework.get_or_create_global_step()

        envir = Corridor(args)
        advcore = TabularRL(learning_rate=1e-2, args=args)
        trainer = a2c.A2CTrainer(envir=envir,
                advcore=advcore,
                tmax=args.batch,
                gamma=config.GAMMA,
                learning_rate=1e-2,
                global_step=global_step)

        saver = tf.train.Saver() # Save everything
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            epoch = 0
            while epoch < total_epoch:
                trainer.train(sess)
                print("===Iteration {}===".format(epoch))
                print('Pol {}'.format(sess.run(advcore.smpolparams)))
                print('Val {}'.format(sess.run(advcore.valparams)))
                epoch += 1

if __name__ == '__main__':
    corridor_main()
