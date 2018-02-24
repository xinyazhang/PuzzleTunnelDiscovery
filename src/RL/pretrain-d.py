'''
    pretrain.py

    Pre-Train the VisionNet and Inverse Model
'''

import tensorflow as tf
import numpy as np
import math
import aniconf12 as aniconf
import matplotlib.pyplot as plt
from scipy.misc import imsave
import matplotlib.animation as animation
import sys
import os
import time
import util
import argparse
import uw_random
import config
import vision
import pyosr
import icm
import threading
import Queue as queue # Python 2, rename to import queue as queue for python 3

MT_VERBOSE = False
# MT_VERBOSE = True
VIEW_CFG = config.VIEW_CFG

def setup_global_variable(args):
    global VIEW_CFG
    if args.ferev in [4,5,6,7,8]:
        VIEW_CFG = config.VIEW_CFG_REV4
    elif args.ferev != 1:
        VIEW_CFG = config.VIEW_CFG_REV2

def _get_action_set(args):
    if args.uniqueaction > 0:
        return [args.uniqueaction]
    return args.actionset

def create_renderer(args):
    view_array = vision.create_view_array_from_config(VIEW_CFG)
    if args.view >= 0:
        view_num = 1
    else:
        view_num = len(view_array)
    w = h = args.res

    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    r = pyosr.Renderer()
    r.pbufferWidth = w
    r.pbufferHeight = h
    r.setup()
    r.loadModelFromFile(aniconf.env_fn)
    r.loadRobotFromFile(aniconf.rob_fn)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    if args.view >= 0:
        r.views = np.array([view_array[args.view]], dtype=np.float32)
    else:
        r.views = np.array(view_array, dtype=np.float32)
    return r

class Animator(object):
    im = None
    keys = None
    failed = False

    def __init__(self, renderer, batch, syncQ=None):
        self.renderer = renderer
        self.keys = []
        self.index = 0
        self.batch = batch
        self.failed = False
        self.syncQ = syncQ

    def buffer_rgb(self):
        index = self.index
        if self.index < len(self.keys):
            return
        if self.syncQ is None:
            self.keys, _ = uw_random.random_path(self.renderer, 0.0125 * 2, self.batch)
        else:
            self.gt = self.syncQ.get()
            self.keys = self.gt.keys
            self.syncQ.task_done()
        self.index = 0

    def get_rgb(self):
        index = self.index
        if index >= len(self.keys):
            self.buffer_rgb()
        if self.syncQ is None:
            return self.render_rgb()
        return self.read_rgb()

    def render_rgb(self):
        index = self.index
        r = self.renderer
        r.state = self.keys[index]
        r.render_mvrgbd()
        # print(r.mvrgb.shape)
        rgb = r.mvrgb.reshape((len(r.views), r.pbufferWidth, r.pbufferHeight, 3))

        print(r.state)
        valid = r.is_valid_state(r.state)
        if not valid:
            print('\tNOT COLLISION FREE, SAN CHECK FAILED')
            self.failed = True
        return rgb

    def read_rgb(self):
        return self.gt.rgb[self.index]

    def perform(self, framedata):
        if self.failed:
            return
        rgb = self.get_rgb()
        rgb = rgb[0] # First View
        if self.im is None:
            print('rgb {}'.format(rgb.shape))
            self.im = plt.imshow(rgb)
        else:
            self.im.set_array(rgb)

        self.index += 1

class GroundTruth:
    pass

def collector(syncQ, sample_num, batch_size, tid, amag, vmag, action_set, args):
    r = create_renderer(args)
    print(r)
    rgb_shape = (len(r.views), r.pbufferWidth, r.pbufferHeight, 3)
    dep_shape = (len(r.views), r.pbufferWidth, r.pbufferHeight, 1)
    if tid == 0:
        last_time = time.time()
    for i in range(sample_num):
        rgbq = []
        depq = []
        if MT_VERBOSE:
            print("!Generating Path #{} by thread {}".format(i, tid))
        if len(action_set) > 0:
            keys, actions = uw_random.random_discrete_path_action_set(r,
                    amag, vmag, batch_size, action_set)
        else:
            keys, actions = uw_random.random_discrete_path(r, amag, vmag, batch_size)
        if MT_VERBOSE:
            print("!Path #{} generated by thread {}".format(i, tid))
        for index in range(len(keys)):
            r.state = keys[index]
            r.render_mvrgbd()
            rgb = r.mvrgb.reshape(rgb_shape)
            dep = r.mvdepth.reshape(dep_shape)
            # rgbq.append(rgb)
            # depq.append(dep)
            rgbq.append(np.copy(rgb))
            depq.append(np.copy(dep))
        if MT_VERBOSE:
            print("!Path #{} rendered by thread {}".format(i, tid))
        gt = GroundTruth()
        gt.actions = np.zeros(shape=(len(actions), 1, uw_random.DISCRETE_ACTION_NUMBER),
                dtype=np.float32)
        for i in range(len(actions)):
            gt.actions[i, 0, actions[i]] = 1.0
        gt_rgb = np.array(rgbq)
        gt_dep = np.array(depq)
        gt.rgb = gt_rgb
        gt.dep = gt_dep
        gt.rgb_1 = gt_rgb[:-1]
        gt.rgb_2 = gt_rgb[1:]
        gt.dep_1 = gt_dep[:-1]
        gt.dep_2 = gt_dep[1:]
        gt.keys = keys
        if MT_VERBOSE:
            print("!GT generated by thread {}".format(tid))
        syncQ.put(gt)
        if MT_VERBOSE:
            print("!GT by thread {} was put into Q".format(tid))
        if tid == 0 and (i+1) % 10 == 0:
            cur_time = time.time()
            print("!GT generation speed: {} samples/sec".format(10/(cur_time - last_time)))
            last_time = cur_time
    print("> GT Thread {} Exits".format(tid))

def spawn_gt_collector_thread(args):
    syncQ = queue.Queue(args.queuemax)
    threads = []
    for i in range(args.threads):
        if MT_VERBOSE:
            print("> action set: {}".format(_get_action_set(args)))
        dic = { 'syncQ' : syncQ, 'sample_num' : args.iter, 'batch_size' :
                args.batch, 'tid' : i, 'amag' : args.amag, 'vmag' : args.vmag,
                'action_set' : _get_action_set(args),
                'args' : args }
        thread = threading.Thread(target=collector, kwargs=dic)
        thread.start()
        threads.append(thread)
    return threads, syncQ

def gt_aggregrate(batching):
    ret = GroundTruth()
    ret.actions = np.concatenate([gt.actions for gt in batching])
    ret.rgb_1 = np.concatenate([gt.rgb[:-1] for gt in batching])
    ret.rgb_2 = np.concatenate([gt.rgb[1:] for gt in batching])
    ret.dep_1 = np.concatenate([gt.dep[:-1] for gt in batching])
    ret.dep_2 = np.concatenate([gt.dep[1:] for gt in batching])
    return ret

def gt_reader(syncQ, args):
    batching = []
    r = None
    for epoch in range(args.total_sample):
        if args.sampletouse > 0:
            idx = epoch % args.sampletouse
        else:
            idx = epoch
        fn = '{}/sample-{}.npz'.format(args.samplein, idx + args.samplebase)
        d = np.load(fn)
        gt = GroundTruth()
        gt.actions = d['A']
        if 'RGB' not in d or args.view >= 0:
            '''
            Render if RGB not present, or a different view is chosen
            '''
            rgbq = []
            depq = []
            if r is None:
                r = create_renderer(args)
                rgb_shape = (len(r.views), r.pbufferWidth, r.pbufferHeight, 3)
                dep_shape = (len(r.views), r.pbufferWidth, r.pbufferHeight, 1)
            keys = d['K']
            nkeys = keys.shape[0]
            for index in range(nkeys):
                r.state = keys[index]
                r.render_mvrgbd()
                rgb = r.mvrgb.reshape(rgb_shape)
                dep = r.mvdepth.reshape(dep_shape)
                rgbq.append(np.copy(rgb))
                depq.append(np.copy(dep))
            gt.rgb = np.array(rgbq)
            gt.dep = np.array(depq)
            gt.keys = keys
        else:
            gt.rgb = d['RGB']
            gt.dep = d['DEP']
        if MT_VERBOSE:
            print("!GT File {} was read".format(fn))
        if args.samplebatching == 1:
            gt.rgb_1 = gt.rgb[:-1]
            gt.rgb_2 = gt.rgb[1:]
            gt.dep_1 = gt.dep[:-1]
            gt.dep_2 = gt.dep[1:]
            syncQ.put(gt)
        else:
            batching.append(gt)
            if len(batching) >= args.samplebatching:
                gt = gt_aggregrate(batching)
                syncQ.put(gt)
                print("!GT RGB_1 {} was queued".format(gt.rgb_1.shape))
                batching = []

def spawn_gt_reader_thread(args):
    syncQ = queue.Queue(args.queuemax)
    dic = {
            'syncQ' : syncQ,
            'args' : args
          }
    thread = threading.Thread(target=gt_reader, kwargs=dic)
    thread.start()
    threads = [thread]
    return threads, syncQ

def save_gt_file(args, gt, epoch):
    fn = '{}/sample-{}'.format(args.sampleout, epoch + args.samplebase)
    imfn = fn+'-peek.png'
    if args.norgbd:
        np.savez(fn, A=gt.actions, K=gt.keys)
    else:
        np.savez(fn, A=gt.actions, RGB=gt.rgb, DEP=gt.dep, K=gt.keys)
    imsave(imfn, gt.rgb_1[0][0])

def pretrain_main(args):
    '''
    CAVEAT: WE MUST CREATE RENDERER BEFORE CALLING ANY TF ROUTINE
    '''
    pyosr.init()
    threads = []
    #total_epoch = args.iter * args.threads
    total_epoch = args.total_epoch

    if args.dryrun:
        r = create_renderer(args)
        fig = plt.figure()
        ra = Animator(r, args.batch)
        ani = animation.FuncAnimation(fig, ra.perform)
        plt.show()
        return
    elif not args.samplein:
        threads, syncQ = spawn_gt_collector_thread(args)
        if args.dryrun2:
            fig = plt.figure()
            ra = Animator(None, args.batch, syncQ)
            ani = animation.FuncAnimation(fig, ra.perform)
            plt.show()
            return
        elif args.dryrun3:
            if args.sampleout:
                for epoch in range(total_epoch):
                    gt = syncQ.get(timeout=60)
                    save_gt_file(args, gt, epoch)
            return
    else:
        threads, syncQ = spawn_gt_reader_thread(args)

    view_array = vision.create_view_array_from_config(VIEW_CFG)
    if args.view >= 0:
        view_num = 1
    else:
        view_num = len(view_array)
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
    util.mkdir_p(ckpt_dir)
    with g.as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        increment_global_step = tf.assign_add(global_step, 1, name='increment_global_step')

        action = tf.placeholder(tf.float32, shape=[None, 1, uw_random.DISCRETE_ACTION_NUMBER])
        rgb_1 = tf.placeholder(tf.float32, shape=[None, view_num, w, h, 3])
        rgb_2 = tf.placeholder(tf.float32, shape=[None, view_num, w, h, 3])
        dep_1 = tf.placeholder(tf.float32, shape=[None, view_num, w, h, 1])
        dep_2 = tf.placeholder(tf.float32, shape=[None, view_num, w, h, 1])
        if args.viewinitckpt:
            model = icm.IntrinsicCuriosityModuleIndependentCommittee(action,
                    rgb_1, dep_1,
                    rgb_2, dep_2,
                    config.SV_VISCFG,
                    config.MV_VISCFG2,
                    args.featnum,
                    args.elu,
                    args.ferev,
                    args.imhidden,
                    args.fehidden,
                    singlesoftmax=args.singlesoftmax)
        elif not args.committee:
            if args.view >= 0:
                with tf.variable_scope(icm.view_scope_name(args.view)):
                    model = icm.IntrinsicCuriosityModule(action,
                            rgb_1, dep_1,
                            rgb_2, dep_2,
                            config.SV_VISCFG,
                            config.MV_VISCFG2,
                            args.featnum,
                            args.elu,
                            args.ferev,
                            args.imhidden,
                            args.fehidden)
                    model.get_inverse_model() # Inverse model also creates variables.
            else:
                model = icm.IntrinsicCuriosityModule(action,
                        rgb_1, dep_1,
                        rgb_2, dep_2,
                        config.SV_VISCFG,
                        config.MV_VISCFG2,
                        args.featnum,
                        args.elu,
                        args.ferev,
                        args.imhidden,
                        args.fehidden)
        else:
            model = icm.IntrinsicCuriosityModuleCommittee(action,
                    rgb_1, dep_1,
                    rgb_2, dep_2,
                    config.SV_VISCFG,
                    config.MV_VISCFG2,
                    args.featnum,
                    args.elu,
                    args.ferev)
        model.get_inverse_model() # Create model.inverse_model_{params,tensor}
        '''
        if args.view >= 0:
            all_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=icm.view_scope_name(args.view))
        else:
            all_params = model.cur_nn_params + model.next_nn_params + model.inverse_model_params
        all_params += [global_step]
        # print(all_params)
        '''
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
        _, predicts = model.get_inverse_model()
        predicts = tf.nn.softmax(predicts)
        loss = model.get_inverse_loss(discrete=True)
        if not args.eval:
            train_op = optimizer.minimize(loss, global_step)
            tf.summary.scalar('loss', loss)
        if args.capture:
            tf.summary.image('input', tf.reshape(tf.slice(rgb_1, [0,0,0,0,0], [1, 1, w, h, 3]), [1, w, h, 3]), 1)
            tf.summary.tensor_summary('predicts', predicts)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(ckpt_dir + '/summary', g)

        saver = tf.train.Saver() # Save everything
        last_time = time.time()
        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            epoch = 0
            accum_epoch = 0
            if args.viewinitckpt:
                model.restore(sess, args.viewinitckpt)
            else:
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
                        print('PANIC: --eval is set but checkpoint does not exits')
                        return
            period_loss = 0.0
            period_accuracy = 0
            total_accuracy = 0
            while epoch < total_epoch:
                gt = syncQ.get(timeout=60)
                dic = {
                        action: gt.actions,
                        rgb_1: gt.rgb_1,
                        rgb_2: gt.rgb_2,
                        dep_1: gt.dep_1,
                        dep_2: gt.dep_2
                      }
                if MT_VERBOSE:
                    print('train with rgb_1 {}'.format(gt.rgb_1.shape))
                batch_size = gt.actions.shape[0]

                if args.sampleout:
                    save_gt_file(args, gt, epoch)
                # print("[{}] Start training".format(epoch))
                if not args.eval:
                    pred, summary, current_loss, _ = sess.run([predicts, summary_op, loss, train_op], feed_dict=dic)
                    train_writer.add_summary(summary, accum_epoch)
                else:
                    current_loss, pred = sess.run([loss, predicts], feed_dict=dic)
                pred_index = np.argmax(pred, axis=2)
                gt_index = np.argmax(gt.actions, axis=2)
                for i in range(gt.actions.shape[0]):
                    delta_accuracy = 1 if pred_index[i, 0] == gt_index[i, 0] else 0
                    period_accuracy += delta_accuracy
                    if delta_accuracy == 0 and args.mispout:
                        samid = epoch + args.samplebase
                        predfn = os.path.join(args.mispout, 'Pred-At-{:07d}'.format(samid))
                        np.savez(predfn, P=pred[i])
                        for V in range(gt.rgb_1.shape[1]):
                            s1 = gt.rgb_1[i,V].reshape(w, h, 3)
                            s2 = gt.rgb_2[i,V].reshape(w, h, 3)
                            s1fn = os.path.join(args.mispout, '{:07d}-IB{:03d}-V{:02d}-1.png'.format(samid, i, V))
                            s2fn = os.path.join(args.mispout, '{:07d}-IB{:03d}-V{:02d}-2.png'.format(samid, i, V))
                            imsave(s1fn, s1)
                            imsave(s2fn, s2)
                        with open(os.path.join(args.mispout, 'Action-At-{:07d}'.format(samid)), 'w') as f:
                            f.write('Action is {}\n'.format(gt_index[i,0]))
                    # print('current preds {} gts {}'.format(pred[i,0], gt.actions[i,0]))
                # print("[{}] End training".format(epoch))
                period_loss += current_loss
                if not args.samplein:
                    syncQ.task_done()
                if (not args.eval) and ((epoch + 1) % 1000 == 0 or time.time() - last_time >= 10 * 60 or epoch + 1 == total_epoch):
                    print("Saving checkpoint")
                    fn = saver.save(sess, ckpt_dir+ckpt_prefix, global_step=global_step)
                    print("Saved checkpoint to {}".format(fn))
                    last_time = time.time()
                if (epoch + 1) % 10 == 0:
                    print("Progress {}/{}".format(epoch, total_epoch))
                    print("Average loss during last 10 iterations: {}".format(period_loss / 10))
                    print("Prediction sample: {}".format(pred[0,0]))
                    print("Action sample: {}".format(gt.actions[0,0]))
                    total_accuracy += period_accuracy
                    p_accuracy_ratio = period_accuracy / (10.0 * batch_size) * 100.0
                    total_accuracy_ratio = total_accuracy / ((epoch+1.0) * batch_size) * 100.0
                    print("Average accuracy during last 10 iterations: {}%. Total: {}%".format(
                        p_accuracy_ratio, total_accuracy_ratio))
                    period_accuracy = 0
                    period_loss = 0
                # print("Epoch {} (Total {}) Done".format(epoch, accum_epoch))
                epoch += 1
                accum_epoch += 1
    total_accuracy += period_accuracy
    total_accuracy_ratio = total_accuracy / ((epoch+1.0) * batch_size) * 100.0
    print("Final Accuracy {}%".format(total_accuracy_ratio))
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckptdir', help='Path for checkpoint files',
            default='ckpt/pretrain-d/')
    parser.add_argument('--sampleout', help='Path to store generated samples',
            default='')
    parser.add_argument('--mispout', help='Path to store mispredicted samples',
            default='')
    parser.add_argument('--samplein', help='Path to load generated samples',
            default='')
    parser.add_argument('--sampletouse',  metavar='NUMBER',
            help='Number of samples to use during the training',
            type=int, default=-1)
    parser.add_argument('--samplebase',  metavar='NUMBER',
            help='Base Number of samples to read/write',
            type=int, default=0)
    parser.add_argument('--uniqueaction',  metavar='NUMBER',
            help='Only generate a specific action when sampling actions',
            type=int, default=-1)
    parser.add_argument('--actionset', nargs='+',
            help='Set to sample actions within, -1 means all actions (0-11)',
            type=int,
            default=[])
    parser.add_argument('--ckptprefix', help='Prefix of checkpoint files',
            default='pretrain-d-ckpt')
    parser.add_argument('--device', help='Prefix of GT file names generated by aa-gt.py',
            default='/gpu:0')
    parser.add_argument('--batch', metavar='NUMBER',
            help='Batch size of each iteration in training',
            type=int, default=32)
    parser.add_argument('--samplebatching', metavar='NUMBER',
            help='Number of samples to aggregrate in training',
            type=int, default=1)
    parser.add_argument('--queuemax', metavar='NUMBER',
            help='Capacity of the synchronized queue to store generated GT',
            type=int, default=32)
    parser.add_argument('--threads', metavar='NUMBER',
            help='Number of GT generation threads',
            type=int, default=1)
    parser.add_argument('--iter', metavar='NUMBER',
            help='Number of samples to generate by each thread',
            type=int, default=0)
    parser.add_argument('--amag', metavar='REAL NUMBER',
            help='Magnitude of discrete actions',
            type=float, default=0.0125 * 4)
    parser.add_argument('--vmag', metavar='REAL NUMBER',
            help='Magnitude of verifying action',
            type=float, default=0.0125 * 4 / 8)
    parser.add_argument('-n', '--dryrun',
            help='Visualize the generated GT without training anything',
            action='store_true')
    parser.add_argument('--dryrun2',
            help='Visualize the generated GT without training anything (MT version)',
            action='store_true')
    parser.add_argument('--dryrun3',
            help='Only generated GT, and store the GT if --sampleout is provided',
            action='store_true')
    parser.add_argument('--elu',
            help='Use ELU instead of ReLU after each NN layer',
            action='store_true')
    parser.add_argument('--singlesoftmax',
            help='Do not apply softmax over member of committee. Hence only one softmax is used to finalize the prediction',
            action='store_true')
    parser.add_argument('--featnum',
            help='Size of the feature vector (aka number of features)',
            type=int, default=256)
    parser.add_argument('--imhidden',
            help='Inverse Model Hidden Layer',
            nargs='+', type=int, default=[])
    parser.add_argument('--fehidden',
            help='Feature Extractor Hidder Layer',
            nargs='+', type=int, default=[1024, 1024])
    parser.add_argument('--eval',
            help='Evaluate the network, rather than training',
            action='store_true')
    parser.add_argument('--continuetrain',
            help='Continue an interrputed training from checkpoint. This basically loads epoch from the checkpoint. WARNING: THIS IS INCOMPATIBLE WITH --samplein',
            action='store_true')
    parser.add_argument('--ferev',
            help='Reversion of Feature Extractor',
            choices=range(1,10+1),
            type=int, default=1)
    parser.add_argument('--capture',
            help='Capture input image to summary',
            action='store_true')
    parser.add_argument('--committee',
            help='Employ a committee of NNs with different weights to extract features/make decisions from different views',
            action='store_true')
    parser.add_argument('--norgbd',
            help='Do not store RGB/D images in storing the sample, to save disk spaces',
            action='store_true')
    parser.add_argument('--view',
            help='Pickup one view to train',
            type=int, default=-1)
    parser.add_argument('--viewinitckpt',
            help='Initialize independent views in sequence with given checkpoints. --eval must present if viewinitckpt is given',
            nargs='*', default=[])
    parser.add_argument('--res',
            help='Resolution',
            type=int, default=config.DEFAULT_RES)

    args = parser.parse_args()
    if (not args.eval) and len(args.viewinitckpt) > 0:
        print('--eval must be set when viewinitckpt is given')
        exit()
    setup_global_variable(args)
    if args.continuetrain:
        if args.samplein:
            print('--continuetrain is incompatible with --samplein')
            exit()
        if args.batching:
            print('--continuetrain is incompatible with --batching')
            exit()
    if -1 in args.actionset:
        args.actionset = [i for i in range(12)]
    if MT_VERBOSE:
        print("Action set {}".format(args.actionset))
    args.total_sample = args.iter * args.threads
    args.total_epoch = args.total_sample / args.samplebatching
    print(args)
    pretrain_main(args)
