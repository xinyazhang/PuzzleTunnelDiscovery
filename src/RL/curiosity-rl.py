'''
    pretrain.py

    Pre-Train the VisionNet and Inverse Model
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
from scipy.misc import imsave # Unused ... now
import time
import uw_random
import config
import pyosr
import rlargs
import a2c
import random
import threading
from six.moves import queue,input
import qtrainer
import ctrainer
import iftrainer
import curiosity
import rlsampler

AlphaPuzzle = curiosity.RigidPuzzle
CuriosityRL = curiosity.CuriosityRL

class TrainingManager:
    kAsyncTask = 1
    kSyncTask = 2
    kExitTask = -1

    def __init__(self, args, g, global_step, batch_normalization):
        '''
        if len(args.egreedy) != 1 and len(args.egreedy) != args.threads:
            assert False,"--egreedy should have only one argument, or match the number of threads"
        '''
        self.args = args
        self.advcore = CuriosityRL(learning_rate=1e-3, args=args, batch_normalization=batch_normalization)
        self.tfgraph = g
        self.threads = []
        self.taskQ = queue.Queue(args.queuemax)
        self.sessQ = queue.Queue(args.queuemax)
        self.reportQ = queue.Queue(args.queuemax)
        self.bnorm = batch_normalization
        if args.train == 'a2c':
            self.trainer = a2c.A2CTrainer(
                    advcore=self.advcore,
                    tmax=args.batch,
                    gamma=config.GAMMA,
                    # gamma=0.5,
                    learning_rate=1e-6,
                    ckpt_dir=args.ckptdir,
                    global_step=global_step,
                    batch_normalization=self.bnorm,
                    period=args.period,
                    LAMBDA=args.LAMBDA)
        elif args.train == 'QwithGT' or args.qlearning_with_gt or args.train == 'QandFCFE':
            self.trainer = qtrainer.QTrainer(
                    advcore=self.advcore,
                    batch=args.batch,
                    learning_rate=1e-4,
                    ckpt_dir=args.ckptdir,
                    period=args.period,
                    global_step=global_step,
                    train_fcfe=(args.train == 'QandFCFE'))
            if args.qlearning_gt_file:
                self.trainer.attach_gt(args.qlearning_gt_file)
        elif args.train == 'curiosity':
            self.trainer = ctrainer.CTrainer(
                    advcore=self.advcore,
                    batch=args.batch,
                    learning_rate=1e-4,
                    ckpt_dir=args.ckptdir,
                    period=args.period,
                    global_step=global_step)
            self.trainer.set_action_set(args.actionset)
            self.trainer.limit_samples_to_use(args.sampletouse)
            if args.samplein != '':
                self.trainer.attach_samplein(args.samplein)
        elif args.train == 'InF':
            # TODO: allow samples from files
            # Note: precomputed samples have one problem:
            #       Actions cannot be translated to new permutations
            self.trainer = iftrainer.IFTrainer(
                    advcore=self.advcore,
                    batch=args.batch,
                    learning_rate=1e-4,
                    ckpt_dir=args.ckptdir,
                    period=args.period,
                    global_step=global_step)
        elif args.train == 'Ionly':
            # SAN Check: Only optimize agains Inverse Model
            # Should work pretty well after loading pretrained weights
            self.trainer = iftrainer.ITrainer(
                    advcore=self.advcore,
                    batch=args.batch,
                    learning_rate=1e-4,
                    ckpt_dir=args.ckptdir,
                    period=args.period,
                    global_step=global_step)
        else:
            assert False, '--train {} not implemented yet'.format(args.train)
        for i in range(args.threads):
            thread = threading.Thread(target=self.run_worker, args=(i,g))
            thread.start()
            self.threads.append(thread)

    def run_worker(self, tid, g):
        '''
        IEnvironment is pre-thread object, mainly due to OpenGL context
        '''
        dpy = pyosr.create_display()
        glctx = pyosr.create_gl_context(dpy)

        with g.as_default():
            args = self.args
            '''
            Disable permutation in TID 0
            Completely disable randomized light source for now
            '''
            # if tid != 0:
            thread_local_envirs = [AlphaPuzzle(args, tid, i) for i in range(args.agents)]
            if args.permutemag >= 0:
                for e in thread_local_envirs:
                    e.enable_perturbation()
            for i,e in enumerate(thread_local_envirs):
                e.egreedy = args.egreedy[i % len(args.egreedy)]
            #else:
                #thread_local_envirs = [AlphaPuzzle(args, tid)]
                # Also disable randomized light position
                # e.r.light_position = uw_random.random_on_sphere(5.0)
            print("[{}] Number of Envirs {}".format(tid, len(thread_local_envirs)))
            while True:
                task = self.taskQ.get()
                if task == self.kExitTask:
                    return 0
                sess = self.sessQ.get()
                '''
                Pickup the envir stochasticly
                '''
                envir = random.choice(thread_local_envirs)
                if not self.args.qlearning_with_gt:
                    print("[{}] Choose Envir with Pertubation {} and egreedy".format(tid, envir.r.perturbation, envir.egreedy))
                self.trainer.train(envir, sess, tid)
                if task == self.kSyncTask:
                    self.reportQ.put(1)

    def train(self, sess, is_async=True):
        self.sessQ.put(sess)
        if is_async:
            self.taskQ.put(self.kAsyncTask)
        else:
            self.taskQ.put(self.kSyncTask)
            done = self.reportQ.get()

    def stop(self):
        for t in self.threads:
            self.taskQ.put(self.kExitTask)
        for t in self.threads:
            t.join()

    def load_pretrain(self, sess, pretrained_ckpt):
        self.advcore.load_pretrain(sess, pretrained_ckpt)

def curiosity_main(args):
    '''
    CAVEAT: WITHOUT ALLOW_GRWTH, WE MUST CREATE RENDERER BEFORE CALLING ANY TF ROUTINE
    '''
    pyosr.init()
    total_epoch = args.total_epoch

    w = h = args.res

    ckpt_dir = args.ckptdir
    ckpt_prefix = args.ckptprefix
    device = args.device

    if 'gpu' in device:
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # session_config = tf.ConfigProto(gpu_options=gpu_options)
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
    else:
        session_config = None

    g = tf.Graph()
    with g.as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        increment_global_step = tf.assign_add(global_step, 1, name='increment_global_step')

        '''
        envir = AlphaPuzzle(args)
        advcore = CuriosityRL(learning_rate=1e-3, args=args)
        trainer = a2c.A2CTrainer(envir=envir,
                advcore=advcore,
                tmax=args.batch,
                gamma=config.GAMMA,
                learning_rate=1e-3,
                global_step=global_step)
        '''
        bnorm = tf.placeholder(tf.bool, shape=()) if args.batchnorm else None
        if args.eval:
            player = rlsampler.create_visualizer(args, g, global_step)
        else:
            trainer = TrainingManager(args, g, global_step, batch_normalization=bnorm)

        # TODO: Summaries

        saver = tf.train.Saver() # Save everything
        last_time = time.time()
        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            epoch = 0
            accum_epoch = 0
            if args.viewinitckpt and not args.eval:
                trainer.load_pretrain(sess, args.viewinitckpt)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
            print('ckpt {}'.format(ckpt))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                accum_epoch = sess.run(global_step)
                print('Restored!, global_step {}'.format(accum_epoch))
                if args.continuetrain:
                    accum_epoch += 1
                    epoch = accum_epoch
            else:
                if args.eval:
                    print('PANIC: --eval is set but checkpoint does not exist')
                    return

            period_loss = 0.0
            period_accuracy = 0
            total_accuracy = 0
            g.finalize() # Prevent accidental changes
            if args.eval:
                player.attach(sess)
                player.play()
                return
            while epoch < total_epoch:
                trainer.train(sess)
                if (not args.eval) and ((epoch + 1) % 1000 == 0 or time.time() - last_time >= 10 * 60 or epoch + 1 == total_epoch):
                    print("Saving checkpoint")
                    fn = saver.save(sess, ckpt_dir+ckpt_prefix, global_step=global_step)
                    print("Saved checkpoint to {}".format(fn))
                    last_time = time.time()
                if (epoch + 1) % 10 == 0:
                    print("Progress {}/{}".format(epoch, total_epoch))
                # print("Epoch {} (Total {}) Done".format(epoch, accum_epoch))
                epoch += 1
                accum_epoch += 1
            trainer.stop() # Must stop before session becomes invalid

if __name__ == '__main__':
    args = rlargs.parse()
    if args.continuetrain:
        if args.samplein:
            print('--continuetrain is incompatible with --samplein')
            exit()
        if args.batching:
            print('--continuetrain is incompatible with --batching')
            exit()
    if -1 in args.actionset:
        args.actionset = [i for i in range(12)]
    args.total_sample = args.iter * args.threads
    args.total_epoch = args.total_sample / args.samplebatching
    print("> Arguments {}".format(args))
    curiosity_main(args)
