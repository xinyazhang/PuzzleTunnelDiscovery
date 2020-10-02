# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
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
import threading

class Corridor(rlenv.IExperienceReplayEnvironment):
    ACTION_DELTA = np.array(
            [[ 1, 0],
             [-1, 0],
             [ 0, 1],
             [ 0,-1]], dtype=np.int32)
    ACTION_SYMBOL = [ '↓', '↑', '→', '←']

    def __init__(self, args, tid=0):
        super(Corridor, self).__init__(tmax=args.batch, erep_cap=128/args.batch)

        self.istate = args.istate
        self.state = np.array(self.istate, dtype=np.int32)
        self.mag = args.res
        self.egreedy = args.egreedy[tid]
        self.noerep = args.noerep
        self.die = args.die
        self.debug = args.debug
        self.israd = args.israd
        self.map = args.map
        w = h = 2 * self.mag + 1
        '''
        Initial reward map: negative rewards on four boundaries
        '''
        self.rmap = np.zeros((2*self.mag + 1, 2*self.mag + 1, 4))
        for i in range(2*self.mag + 1):
            self.rmap[    0, i, 1] = -1.0
            self.rmap[w - 1, i, 0] = -1.0
            self.rmap[i,     0, 3] = -1.0
            self.rmap[i, h - 1, 2] = -1.0
        if self.map == 0:
            for i in range(w):
                self.rmap[i, 0, 3] = 1.0
                self.rmap[i, h - 1, 2] = 1.0
        elif self.map == 1:
            self.rmap[0,0,3] = 1.0
        elif self.map == 2:
            # Add barrier
            for i in range(2*self.mag):
                self.rmap[1,i,1] = -1.0
                self.rmap[0,i,0] = -1.0
            print(self.rmap)
            # Reward for leaving (two directions)
            self.rmap[0,0,3] = 100.0
            self.rmap[0,0,1] = 100.0
            if len(args.ig) > 0:
                for x,y,a in zip(args.ig[0::3], args.ig[1::3], args.ig[2::3]):
                    self.rmap[x,y,a] = 1.0
        elif self.map == 3:
            # Snake like barrier
            ## First: all barriers
            for x in range(w-1):
                for y in range(h):
                    self.rmap[x+1,y,1] = -1.0
                    self.rmap[x,y,0] = -1.0
            ## Second: add holes
            for x in range(w-1):
                if x % 2 == 0:
                    self.rmap[x+1,-1,1] = 0.0
                    self.rmap[x,-1,0] = 0.0
                else:
                    self.rmap[x+1,0,1] = 0.0
                    self.rmap[x,0,0] = 0.0
            print(self.rmap)
            # Reward for leaving (two directions)
            self.rmap[0,0,3] = 100.0
            self.rmap[0,0,1] = 100.0
            if len(args.ig) > 0:
                for x,y,a in zip(args.ig[0::3], args.ig[1::3], args.ig[2::3]):
                    self.rmap[x,y,a] = 1.0
        self.vgt = None

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

    def peek_act(self, action, pprefix=''):
        nstate = self.state + self.ACTION_DELTA[action]
        reward = 0
        reaching_terminal = False
        ratio = 1.0
        if self.map > 0:
            x,y = self.state + self.mag
            reward = self.rmap[x,y,action]
            if reward > 0:
                reaching_terminal = True
            elif reward < 0:
                ratio = 0.0
                reaching_terminal = self.die
                nstate = np.copy(self.state)
        else:
            if abs(nstate[1]) > self.mag: # Y boundaries
                reward = 1.0
                reaching_terminal = True
            if abs(nstate[0]) > self.mag: # X boundaries
                reward = -1.0
                ratio = 0.0
                reaching_terminal = self.die
        np.clip(nstate, -self.mag, self.mag, nstate)
        if self.debug:
            print('> state {} action {} nstate {} r {}'.format(self.state, action, nstate, reward))
        return nstate, reward, reaching_terminal, ratio

    def reset(self):
        super(Corridor, self).reset()
        self.state = np.array(self.istate, dtype=np.int32)
        self.state += np.random.randint(-self.israd, self.israd+1, size=2, dtype=np.int32)
        np.clip(self.state, -self.mag, self.mag, self.state)

    '''
    This disables experience replay
    '''
    def sample_in_erep(self, pprefix=''):
        if self.noerep:
            return [],0,0,0
        return super(Corridor, self).sample_in_erep(pprefix)

    def get_value_gt(self):
        if self.vgt is not None:
            return self.vgt
        w = h = 2*self.mag + 1
        self.vgt = np.zeros((w, h))
        """
        '''
        Debug: half 0 half 1
        '''
        for x in range(w):
            for y in range(h):
                if x > self.mag:
                    self.vgt[x,y] = 1
        return self.vgt
        """
        q = []
        for x in range(w):
            for y in range(h):
                for a in range(len(self.ACTION_DELTA)):
                    r = self.rmap[x,y,a]
                    if r > 0:
                        q.append((x,y,r))
                        self.vgt[x,y] = r
                        break
        while len(q) > 0:
            x,y,r = q.pop(0)
            print("{} {} r {}".format(x,y,r))
            for a in range(len(self.ACTION_DELTA)):
                if self.rmap[x,y,a] < 0:
                    continue
                nstate = np.array([x,y]) + self.ACTION_DELTA[a]
                np.clip(nstate, 0, w, nstate)
                nx,ny = nstate
                if self.vgt[nx, ny] > 0:
                    continue
                # nr = r - 0.01
                nr = r * config.GAMMA
                self.vgt[nx,ny] = nr
                q.append((nx,ny,nr))
        return self.vgt

    def get_perturbation(self):
        return None

class TabularRL(rlenv.IAdvantageCore):

    using_lstm = False
    action_space_dimension = 4

    def __init__(self, learning_rate, args, master=None, shadowid=-1):
        super(TabularRL, self).__init__()
        self.h = self.w = args.res * 2 + 1 # [-res, -res] to [res, res]
        self.mani = args.mani

        self.action_tensor = tf.placeholder(tf.float32, shape=[None, 1, 4], name='ActionPh')
        self.rgb_1_tensor = tf.placeholder(tf.int32, shape=[None, 2], name='Rgb1Ph')
        self.dep_1_tensor = tf.placeholder(tf.int32, shape=[None], name='Dep1Ph')
        self.rgb_2_tensor = tf.placeholder(tf.int32, shape=[None, 2], name='Rgb2Ph')
        self.dep_2_tensor = tf.placeholder(tf.int32, shape=[None], name='Dep2Ph')

        if shadowid >= 0:
            vprefix = 'Shadow{}_'.format(shadowid)
        else:
            vprefix = ''

        frgb_1 = tf.cast(self.rgb_1_tensor, tf.float32)
        frgb_2 = tf.cast(self.rgb_2_tensor, tf.float32)

        if not self.mani:
            self.polparams = tf.get_variable(vprefix+'polgrid', shape=[self.w, self.h, 4],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            self.valparams = tf.get_variable(vprefix+'valgrid', shape=[self.w, self.h],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            self.smpolparams = tf.nn.softmax(logits=self.polparams)

            self.indices = tf.reshape(self.rgb_1_tensor, [-1, 2]) + args.res
            self.polout = tf.gather_nd(self.smpolparams,
                indices=self.indices)
            print('polout shape {}'.format(self.polout.shape))
            self.valout = tf.gather_nd(self.valparams,
                indices=self.indices)
            self.smpol = tf.reshape(self.policy, [-1,1,4])
        else:
            febase = [64, 64]
            featnums = febase + [4]
            self.pol_applier = vision.ConvApplier(None, featnums, vprefix+'PolMani', elu=True, nolu_at_final=True)
            self.polparams, self.polout = self.pol_applier.infer(frgb_1)
            featnums = febase + [1]
            self.val_applier = vision.ConvApplier(None, featnums, vprefix+'ValMani', elu=True, nolu_at_final=True)
            self.valparams, self.valout = self.val_applier.infer(frgb_1)
            self.smpol = tf.nn.softmax(tf.reshape(self.policy, [-1,1,4]))
            rgb_list = []
            for x in range(-args.res, args.res+1):
                for y in range(-args.res, args.res+1):
                    rgb_list.append([x,y])
            self.pvgrid = np.array(rgb_list, dtype=np.int32)

        atensor = tf.reshape(self.action_tensor, [-1, 4])
        fwd_input = tf.concat([atensor, frgb_1], 1)
        featnums = [64,2]
        self.forward_applier = vision.ConvApplier(None, featnums, vprefix+'ForwardModelNet', elu=True, nolu_at_final=True)
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
        self.debug = args.debug
        self.cr = args.cr
        self.master = master

        ws = self.get_weights()
        if master is not None:
            mws = self.master.get_weights()
            self.download_op = [tf.assign(target,source) for target,source in zip(ws, mws)]
        else:
            self.grads_in1 = [tf.placeholder(tensor.dtype, shape=tensor.shape) for tensor in ws]
            self.grads_in2 = [tf.placeholder(tensor.dtype, shape=tensor.shape) for tensor in ws]
            self.grads_op = [tf.assign_add(var, in2 - in1) for var,in1,in2 in zip(ws, self.grads_in1, self.grads_in2)]
            # self.upload_op = [tf.assign(target,source) for target,source in zip(ws, mws)]

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

    def polmap(self, sess):
        if not self.mani:
            return sess.run(self.smpolparams)
        print(sess.run(self.softmax_policy, feed_dict={ self.rgb_1: [[-8,0]] }))
        poldata = sess.run(self.softmax_policy, feed_dict={ self.rgb_1: self.pvgrid })
        return np.reshape(poldata, (self.w, self.h, 4))
        # poldata = sess.run(self.softmax_policy, feed_dict={ self.rgb_1: [[-8,0]] })
        # return np.reshape(poldata, (1, 1, 4))

    def valmap(self, sess):
        if not self.mani:
            return sess.run(self.valparams)
        valdata = sess.run(self.value, feed_dict={ self.rgb_1: self.pvgrid })
        return np.reshape(valdata, (self.w, self.h))

    def evaluate(self, vstates, sess, tensors, additional_dict=None):
        rgbs = [state[0] for state in vstates]
        dic = {
                self.rgb_1 : rgbs,
        }
        if additional_dict is not None:
            dic.update(additional_dict)
        return sess.run(tensors, feed_dict=dic)

    def make_decision(self, envir, policy_dist, pprefix=''):
        best = np.argmax(policy_dist, axis=-1)
        if random.random() > envir.egreedy:
            ret = random.randrange(self.action_space_dimension)
        else:
            ret = np.asscalar(best)
        if self.debug:
            print('{}Action best {} chosen {}'.format(pprefix, best, ret))
        return ret

    def get_artificial_reward(self, envir, sess, state_1, adist, state_2, pprefix=''):
        if not self.cr:
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
        if not self.cr:
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
        if not self.cr:
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

    def get_weights(self):
        if self.mani:
            return self.polparams + self.valparams + self.forward_params

        return [self.polparams, self.valparams] + self.forward_params

    def upload(self, sess):
        if self.master is None:
            return
        period_end_weights = sess.run(self.get_weights())
        for op,in1,in2,old,new in zip(self.master.grads_op, self.master.grads_in1, self.master.grads_in2, self.period_init_weights, period_end_weights):
            sess.run(op, feed_dict={in1 : old, in2: new})

    def download(self, sess):
        if self.master is None:
            return
        sess.run(self.download_op)
        self.period_init_weights = sess.run(self.master.get_weights())

    def get_gt_loss(self, envir):
        itensor = tf.placeholder(tf.float32, shape=[None], name='GtPh')
        vtensor = tf.reshape(self.value, shape=[-1])
        loss = tf.nn.l2_loss(itensor-vtensor)
        print("itensor {} self.value {}".format(itensor.shape, vtensor.shape))
        return itensor, loss

def run_trainer(args, sess, envir, advcore, trainer, saver, gs, tid):
    total_epoch = args.iter
    epoch = 0
    period = 0
    advcore.download(sess)
    while epoch < total_epoch:
        trainer.train(envir, sess)
        if args.debug:
            np.set_printoptions(precision=4, linewidth=180)
            allval = sess.run(advcore.valmap)
            print("===Iteration {}===".format(epoch))
            print('Pol {}'.format(advcore.polmap(sess)))
            print('Val\n{}'.format(allval))
            np.set_printoptions()
        epoch += 1
        period += 1
        if period >= args.period:
            advcore.upload(sess)
            advcore.download(sess)
            period = 0
        if tid == 0 and epoch % 1024 == 0:
            fn = saver.save(sess, args.ckpt, global_step=gs)
            print("Saved checkpoint to {}".format(fn))

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
    parser.add_argument('--showv',
            help='Show value',
            action='store_true')
    parser.add_argument('--showgt',
            help='Show GT value',
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
    parser.add_argument('--mani',
            help='Use NN instead of tabular to represent the polnet and valnet',
            action='store_true')
    parser.add_argument('--traingt',
            help='Train the Value net directly with caculated GT',
            action='store_true')
    parser.add_argument('--debug',
            help='Enable debugging output',
            action='store_true')
    parser.add_argument('--threads',
            type=int,
            help='Number of threads',
            default=1)
    parser.add_argument('--egreedy',
            type=float,
            nargs='+',
            help='epsilon-greedy',
            default=[0.5])
    parser.add_argument('--israd',
            type=int,
            help='Initial State RADius',
            default=0)
    parser.add_argument('--period',
            type=int,
            help='Weight Synchronization Period (in number of batches)',
            default=0)
    parser.add_argument('--cr',
            help='Enable CuRiosity',
            action='store_true')
    parser.add_argument('--map',
            type=int,
            choices=[0,1,2,3],
            help='Select Map',
            default=0)
    parser.add_argument('--istate',
            type=int,
            nargs=2,
            help='Set the starting point',
            default=[0,0])
    parser.add_argument('--ig',
            type=int,
            nargs='*',
            help='Intermediate Goal (in tuple of x,y,a)',
            default=[])

    args = parser.parse_args()
    assert len(args.egreedy) == args.threads, "--egreedy should match --thread"
    assert args.mani if args.traingt else True, "--traingt must be used with --mani"

    if args.mani:
        lr = 1e-3
    else:
        lr = 1e-2

    # g = tf.Graph()
    #with g.as_default(), tf.device("/cpu:0"):
    with tf.device("/cpu:0"):
        global_step = tf.contrib.framework.get_or_create_global_step()

        envirs = [Corridor(args, tid) for tid in range(args.threads)]
        if args.showgt:
            np.set_printoptions(precision=4, linewidth=240)
            print(envirs[0].get_value_gt())
            return
        # "Master" weights
        advcore = TabularRL(learning_rate=lr, args=args)
        advcores = [TabularRL(learning_rate=lr, args=args, master=advcore, shadowid=tid) for tid in range(args.threads)]
        trainers = [a2c.A2CTrainer(
                advcore=advcores[i],
                tmax=args.batch,
                gamma=config.GAMMA,
                learning_rate=lr,
                ckpt_dir=None,
                global_step=global_step,
                debug=args.debug) for i in range(args.threads)]
        if args.traingt:
            envir = envirs[0]
            gt_optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
            gt_input,gt_loss = advcore.get_gt_loss(envir)
            gt_train_op = gt_optimizer.minimize(gt_loss, global_step=global_step)
            gv = np.reshape(envir.get_value_gt(), (-1))

        saver = tf.train.Saver(advcore.get_weights()) # Only save/load weigths in the master
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # print(sess.run(advcore.valparams))
            epoch = 0

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=args.ckpt)
            print('ckpt {}'.format(ckpt))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            if args.showp:
                envir = envirs[0]
                p = advcore.polmap(sess)
                np.set_printoptions(precision=3, linewidth=180)
                print(p[:,:,0])
                print(p[:,:,1])
                print(p[:,:,2])
                print(p[:,:,3])
                pmax = np.argmax(p, axis=2)
                print(pmax)
                vp = [[envir.ACTION_SYMBOL[pmax[y][x]] for x in range(2*args.res+1)] for y in range(2*args.res+1)]
                for row in vp:
                    for e in row:
                        print(e, end=' ')
                    print('')
                # print(sess.run(advcore.polparams))
                return
            elif args.showv:
                np.set_printoptions(precision=4, linewidth=180)
                print(advcore.valmap(sess))
                return
            elif args.play:
                envir = envirs[0]
                envir.egreedy = 0.99
                while True:
                    pol = advcore.evaluate([envir.vstate], sess, [advcore.softmax_policy])
                    pol = pol[0][0]
                    a = advcore.make_decision(envir, pol, '')
                    print("[debug] a {}".format(a))
                    ns, r, rt, ratio = envir.peek_act(a,'')
                    print("state {} action {} next {} reward {}".format(envir.state, a, ns, r))
                    envir.state = ns
                    if rt:
                        break
                return
            elif args.traingt:
                assert args.mani, "--mani should be specified with --traingt"
                np.set_printoptions(precision=3, linewidth=240, suppress=True)
                print("Init Error\n{}".format(advcore.valmap(sess) - envir.get_value_gt()))
                # print(advcore.pvgrid)
                # print(sess.run(advcore.valparams))
                # print(advcore.valmap(sess))
                # print(gv)
                for epoch in range(args.iter):
                    fd = {gt_input:gv, advcore.rgb_1:advcore.pvgrid}
                    sess.run(gt_train_op, feed_dict=fd)
                    # print(advcore.valmap(sess))
                    if epoch % 1024 == 1024 - 1:
                        fn = saver.save(sess, args.ckpt, global_step=global_step)
                        print("Saved checkpoint to {}".format(fn))
                        # print(advcore.valmap(sess))
                        # print(advcore.valmap(sess))
                        print("Current Error\n{}".format(advcore.valmap(sess) - envir.get_value_gt()))
                        print("Loss {}".format(sess.run(gt_loss,feed_dict=fd)))
                print("Final Error\n{}".format(advcore.valmap(sess) - envir.get_value_gt()))
                saver.save(sess, args.ckpt, global_step=global_step)
                return
            threads = []
            for i in range(args.threads):
                a=(args, sess, envirs[i], advcores[i], trainers[i], saver, global_step, i)
                thread = threading.Thread(target=run_trainer, args=a)
                thread.start()
                threads.append(thread)
            for t in threads:
                t.join()
            saver.save(sess, args.ckpt, global_step=global_step)

if __name__ == '__main__':
    corridor_main()
