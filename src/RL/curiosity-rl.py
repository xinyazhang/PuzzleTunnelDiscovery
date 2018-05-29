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
from rlreanimator import reanimate
import qtrainer
import ctrainer
import curiosity

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
        elif args.train == 'QwithGT' or args.qlearning_with_gt:
            self.trainer = qtrainer.QTrainer(
                    advcore=self.advcore,
                    batch=args.batch,
                    learning_rate=1e-4,
                    ckpt_dir=args.ckptdir,
                    period=args.period,
                    global_step=global_step)
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
        else:
            assert False, '--train {} not implemented yet'.format(args.train)
        assert not (self.advcore.using_lstm and self.trainer.erep_sample_cap > 0), "CuriosityRL does not support Experience Replay with LSTM"
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
            for e in thread_local_envirs[1:]:
                e.enable_perturbation()
            for i in range(1, len(thread_local_envirs)):
                thread_local_envirs[i].egreedy = args.egreedy[i % len(args.egreedy)]
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

class RLVisualizer(object):
    def __init__(self, args, g, global_step):
        self.args = args
        self.dpy = pyosr.create_display()
        self.ctx = pyosr.create_gl_context(self.dpy)
        self.envir = AlphaPuzzle(args, 0)
        self.envir.egreedy = 0.995
        self.advcore = CuriosityRL(learning_rate=1e-3, args=args)
        self.advcore.softmax_policy # Create the tensor
        self.gview = 0 if args.obview < 0 else args.obview
        self.envir.enable_perturbation()

    def attach(self, sess):
        self.sess = sess

class PolicyPlayer(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(PolicyPlayer, self).__init__(args, g, global_step)

    def play(self):
        reanimate(self)

    def __iter__(self):
        envir = self.envir
        sess = self.sess
        advcore = self.advcore
        reaching_terminal = False
        pprefix = "[0] "
        while True:
            rgb,_ = envir.vstate
            yield rgb[self.gview] # First view
            if reaching_terminal:
                print("##########CONGRATS TERMINAL REACHED##########")
                envir.reset()
            policy = advcore.evaluate([envir.vstate], sess, [advcore.softmax_policy])
            policy = policy[0][0]
            action = advcore.make_decision(envir, policy, pprefix)
            print("PolicyPlayer pol {}".format(policy))
            print("PolicyPlayer Action {}".format(action))
            nstate,reward,reaching_terminal,ratio = envir.peek_act(action, pprefix=pprefix)
            envir.qstate = nstate

class QPlayer(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(QPlayer, self).__init__(args, g, global_step)
        if args.permutemag > 0:
            self.envir.enable_perturbation()

    def render(self, envir, state):
        envir.qstate = state
        return envir.vstate

    def play(self):
        if self.args.sampleout:
            self._sample()
        else:
            self._play()

    def _sample(self):
        Q = [] # list of states
        V = [] # list of numpy array of batched values
        args = self.args
        sess = self.sess
        advcore = self.advcore
        envir = self.envir
        assert args.iter % args.batch == 0, "presumably --iter is dividable by --batch"
        for i in range(args.iter/args.batch):
            states = [uw_random.gen_unit_init_state(envir.r) for i in range(args.batch)]
            Q += states
            images = [self.render(envir, state) for state in states]
            batch_rgb = [image[0] for image in images]
            batch_dep = [image[1] for image in images]
            dic = {
                    advcore.rgb_1: batch_rgb,
                    advcore.dep_1: batch_dep,
                  }
            values = sess.run(advcore.value, feed_dict=dic)
            values = np.reshape(values, [-1]) # flatten
            V.append(values)
        Q = np.array(Q)
        V = np.concatenate(V)
        np.savez(args.sampleout, Q=Q, V=V)

    def _play(self):
        reanimate(self)

    def __iter__(self):
        args = self.args
        sess = self.sess
        advcore = self.advcore
        envir = self.envir
        envir.enable_perturbation()
        envir.reset()
        current_value = -1
        TRAJ = []
        while True:
            TRAJ.append(envir.qstate)
            yield envir.vstate[0][args.obview] # Only RGB
            NS = []
            images = []
            # R = []
            T = []
            TAU = []
            state = envir.qstate
            print("> Current State {}".format(state))
            for action in range(uw_random.DISCRETE_ACTION_NUMBER):
                envir.qstate = state # IMPORTANT: Restore the state to unpeeked condition
                nstate, reward, terminal, ratio = envir.peek_act(action)
                envir.qstate = nstate
                NS.append(nstate)
                T.append(terminal)
                TAU.append(ratio)
                image = envir.vstate
                images.append(image)
            batch_rgb = [image[0] for image in images]
            batch_dep = [image[1] for image in images]
            dic = {
                    advcore.rgb_1: batch_rgb,
                    advcore.dep_1: batch_dep,
                  }
            values = sess.run(advcore.value, feed_dict=dic)
            values = np.reshape(values, [-1]) # flatten
            best = np.argmax(values, axis=0)
            print("> Current Values {}".format(values))
            print("> Taking Action {} RATIO {}".format(best, TAU[best]))
            print("> NEXT State {} Value".format(NS[best], values[best]))
            envir.qstate = NS[best]
            should_reset = False
            if current_value > values[best] or TAU[best] == 0.0:
                input("FATAL: Hit Local Maximal! Press Enter to restart")
                should_reset = True
            else:
                current_value = values[best]
            if T[best]:
                input("DONE! Press Enter to restart ")
                should_reset = True
            if should_reset:
                fn = input("Enter the filename to save the trajectory ")
                if fn:
                    TRAJ.append(envir.qstate)
                    TRAJ = np.array(TRAJ)
                    np.savez(fn, TRAJ=TRAJ, SINGLE_PERM=envir.get_perturbation())
                envir.reset()
                current_value = -1
                TRAJ = []

class CuriositySampler(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(CuriositySampler, self).__init__(args, g, global_step)
        assert args.visualize == 'curiosity', '--visualize must be curiosity'
        assert args.curiosity_type == 1, "--curiosity_type should be 1 if --visualize is enabled"
        assert args.sampleout != '', '--sampleout must be enabled for --visualize curiosity'

    def play(self):
        args = self.args
        sess = self.sess
        advcore = self.advcore
        envir = self.envir
        samples= []
        curiosities_by_action = [ [] for i in range(uw_random.DISCRETE_ACTION_NUMBER) ]
        for i in range(args.iter):
            state = uw_random.gen_unit_init_state(envir.r)
            envir.qstate = state
            samples.append(state)
            for action in range(uw_random.DISCRETE_ACTION_NUMBER):
                nstate, reward, terminal, ratio = envir.peek_act(action)
                areward = advcore.get_artificial_reward(envir, sess,
                        state, action, nstate, ratio)
                curiosities_by_action[action].append(areward)
        samples = np.array(samples)
        curiosity = np.array(curiosities_by_action)
        np.savez(args.sampleout, Q=samples, C=curiosity)

def create_visualizer(args, g, global_step):
    if args.qlearning_with_gt:
        # assert args.sampleout, "--sampleout is required to store the samples for --qlearning_with_gt"
        assert args.iter > 0, "--iter needs to be specified as the samples to generate"
        # assert False, "Evaluating of Q Learning is not implemented yet"
        return QPlayer(args, g, global_step)
    elif args.visualize == 'policy':
        return PolicyPlayer(args, g, global_step)
    elif args.visualize == 'curiosity':
        return CuriositySampler(args, g, global_step)
    assert False, '--visualize {} is not implemented yet'.format(args.visualize)

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
            player = create_visualizer(args, g, global_step)
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
