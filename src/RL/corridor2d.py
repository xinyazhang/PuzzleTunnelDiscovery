# -*- coding: utf-8 -*-

from __future__ import print_function
import rlenv
import tensorflow as tf
import argparse
import config
import a2c
import numpy as np
import random
import vision

class Corridor(rlenv.IExperienceReplayEnvironment):
    ACTION_DELTA = np.array(
            [[ 1, 0],
             [-1, 0],
             [ 0, 1],
             [ 0,-1]], dtype=np.int32)
    ACTION_SYMBOL = [ '↓', '↑', '→', '←']

    def __init__(self, args):
        super(Corridor, self).__init__(tmax=args.batch, erep_cap=4)

        self.state = np.array([0,0], dtype=np.int32)
        self.mag = args.res
        self.egreedy = 0.5
        self.noerep = args.noerep
        self.die = args.die

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

    def peek_act(self, action, pprefix):
        nstate = self.state + self.ACTION_DELTA[action]
        reward = 0
        reaching_terminal = False
        ratio = 1.0
        if abs(nstate[1]) > self.mag: # Y boundaries
            reward = 1.0
            reaching_terminal = True
        if abs(nstate[0]) > self.mag: # X boundaries
            reward = -1.0
            ratio = 0.0
            reaching_terminal = self.die
        np.clip(nstate, -self.mag, self.mag, nstate)
        print('> state {} action {} nstate {} r {}'.format(self.state, action, nstate, reward))
        return nstate, reward, reaching_terminal, ratio

    def reset(self):
        super(Corridor, self).reset()
        self.state = np.array([0,0], dtype=np.int32)

    '''
    This disables experience replay
    '''
    def sample_in_erep(self, pprefix):
        if self.noerep:
            return [],0,0,0
        return super(Corridor, self).sample_in_erep(pprefix)

class TabularRL(rlenv.IAdvantageCore):

    using_lstm = False
    action_space_dimension = 4

    def __init__(self, learning_rate, args):
        super(TabularRL, self).__init__()
        self.h = self.w = args.res * 2 + 1 # [-res, -res] to [res, res]

        self.action_tensor = tf.placeholder(tf.float32, shape=[None, 1, 4], name='ActionPh')
        self.rgb_1_tensor = tf.placeholder(tf.int32, shape=[None, 2], name='Rgb1Ph')
        self.dep_1_tensor = tf.placeholder(tf.int32, shape=[None], name='Dep1Ph')
        self.rgb_2_tensor = tf.placeholder(tf.int32, shape=[None, 2], name='Rgb2Ph')
        self.dep_2_tensor = tf.placeholder(tf.int32, shape=[None], name='Dep2Ph')

        self.polparams = tf.get_variable('polgrid', shape=[self.w, self.h, 4],
                dtype=tf.float32, initializer=tf.zeros_initializer())
        self.valparams = tf.get_variable('valgrid', shape=[self.w, self.h],
                dtype=tf.float32, initializer=tf.zeros_initializer())
        self.smpolparams = tf.nn.softmax(logits=self.polparams)

        self.indices = tf.reshape(self.rgb_1_tensor, [-1, 2]) + args.res
        self.polout = tf.gather_nd(self.smpolparams,
            indices=self.indices)
        print('polout shape {}'.format(self.polout.shape))
        self.valout = tf.gather_nd(self.valparams,
            indices=self.indices)
        self.smpol = tf.reshape(self.policy, [-1,1,4])

        atensor = tf.reshape(self.action_tensor, [-1, 4])
        frgb_1 = tf.cast(self.rgb_1_tensor, tf.float32)
        frgb_2 = tf.cast(self.rgb_2_tensor, tf.float32)
        fwd_input = tf.concat([atensor, frgb_1], 1)
        featnums = [64,2]
        self.forward_applier = vision.ConvApplier(None, featnums, 'ForwardModelNet', elu=True, nolu_at_final=True)
        self.forward_params, self.forward = self.forward_applier.infer(fwd_input)
        # Per action curiosity
        print('FWD {}'.format(self.forward.shape))
        print('FRGB_2 {}'.format(frgb_2.shape))
        self.sqdiff = tf.squared_difference(self.forward, frgb_2)
        print('SQDIFF {}'.format(self.sqdiff.shape))
        self.curiosity = tf.reduce_mean(self.sqdiff,
                                   axis=[1], keepdims=False)

        print('> Polout {} Valout {} Curiosity {}'.format(self.polout.shape, self.valout.shape, self.curiosity.shape))
        self.loss = None

    @property
    def softmax_policy(self):
        return self.smpol

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

    def evaluate(self, vstates, sess, tensors, additional_dict=None):
        rgbs = [state[0] for state in vstates]
        dic = {
                self.rgb_1 : rgbs,
        }
        if additional_dict is not None:
            dic.update(additional_dict)
        return sess.run(tensors, feed_dict=dic)

    def make_decision(self, envir, policy_dist, pprefix):
        best = np.argmax(policy_dist, axis=-1)
        if random.random() < envir.egreedy:
            ret = random.randrange(self.action_space_dimension)
        else:
            ret = np.asscalar(best)
        print('{}Action best {} chosen {}'.format(pprefix, best, ret))
        return ret

    def get_artificial_reward(self, envir, sess, state_1, adist, state_2, pprefix):
        return 0
        envir.qstate = state_1
        vs1 = envir.vstate
        envir.qstate = state_2
        vs2 = envir.vstate
        # print('state_1 {}'.format(state_1))
        # print('state_2 {}'.format(state_2))
        # print('vs1 {}'.format(vs1))
        # print('vs2 {}'.format(vs2))
        dic = {
                self.action_tensor : [[adist]], # Expand from [12] (A only) to [1,1,12] ([F,V,A])
                self.rgb_1 : [vs1[0]],
                self.rgb_2 : [vs2[0]],
              }
        ret = sess.run(self.curiosity, feed_dict=dic)[0]
        return ret

    def get_artificial_from_experience(self, sess, vstates, actions):
        return [0] * len(actions)
        adists_array = []
        for ai in actions:
            adist = np.zeros(shape=(1, self.action_space_dimension),
                    dtype=np.float32)
            adist[0, ai] = 1.0
            adists_array.append(adist)
        rgbs = [state[0] for state in vstates]
        dic = {
                self.action_tensor : adists_array,
                self.rgb_1 : rgbs[:-1],
                self.rgb_2 : rgbs[1:],
              }
        ret = sess.run(self.curiosity, feed_dict=dic)
        return ret

    def build_loss(self):
        return 0
        if self.loss is not None:
            return self.loss
        # Sum of all curiosities
        self.loss = tf.reduce_sum(self.curiosity)
        return self.loss

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
    parser.add_argument('--ckpt', metavar='DIR',
            help='Check points prefix',
            default='ackpt/corridor2/')
    parser.add_argument('--showp',
            help='Show policy',
            action='store_true')
    parser.add_argument('--noerep',
            help='Disable Experience Replay',
            action='store_true')
    parser.add_argument('--play',
            help='Play',
            action='store_true')
    parser.add_argument('--die',
            help='Restart after hitting boundary',
            action='store_true')
    parser.add_argument('--debug',
            help='Enable debugging output',
            action='store_true')

    args = parser.parse_args()

    total_epoch = args.iter

    g = tf.Graph()
    with g.as_default(), tf.device("/cpu:0"):
        global_step = tf.contrib.framework.get_or_create_global_step()

        lr = 1e-2
        envir = Corridor(args)
        advcore = TabularRL(learning_rate=lr, args=args)
        trainer = a2c.A2CTrainer(
                advcore=advcore,
                tmax=args.batch,
                gamma=config.GAMMA,
                learning_rate=lr,
                ckpt_dir=None,
                global_step=global_step,
                debug=args.debug)

        saver = tf.train.Saver() # Save everything
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            epoch = 0

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=args.ckpt)
            print('ckpt {}'.format(ckpt))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            if args.showp:
                p = sess.run(advcore.polparams)
                pmax = np.argmax(p, axis=2)
                print(pmax)
                vp = [[envir.ACTION_SYMBOL[pmax[y][x]] for x in range(2*args.res+1)] for y in range(2*args.res+1)]
                for row in vp:
                    for e in row:
                        print(e, end=' ')
                    print('')
                return
            elif args.play:
                envir.egreedy = 1 - 0.99
                while True:
                    pol = advcore.evaluate([envir.vstate], sess, [advcore.softmax_policy])
                    pol = pol[0][0]
                    a = advcore.make_decision(envir, pol, '')
                    print("[debug] a {}".format(a))
                    ns, r, rt, ratio = envir.peek_act(a,'')
                    print("state {} action {} next {}".format(envir.state, a, ns))
                    envir.state = ns
                    if rt:
                        break
                return
            while epoch < total_epoch:
                trainer.train(envir, sess)
                if args.debug:
                    np.set_printoptions(precision=4, linewidth=180)
                    allval = sess.run(advcore.valparams)
                    print("===Iteration {}===".format(epoch))
                    print('Pol {}'.format(sess.run(advcore.smpolparams)))
                    print('Val\n{}'.format(allval))
                    np.set_printoptions()
                epoch += 1
                if epoch % 1024 == 0:
                    fn = saver.save(sess, args.ckpt, global_step=global_step)
                    print("Saved checkpoint to {}".format(fn))
            saver.save(sess, args.ckpt, global_step=global_step)

if __name__ == '__main__':
    corridor_main()
