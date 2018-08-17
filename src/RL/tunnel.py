from __future__ import print_function
import numpy as np
import tensorflow as tf
import pyosr
import rlutil
import vision
import config
import curiosity
import uw_random

class TunnelFinderCore(object):

    def __init__(self, learning_rate, args, batch_normalization=None):
        self.view_num, self.views = rlutil.get_view_cfg(args)
        self.args = args
        self.batch_normalization = batch_normalization

        self.action_space_dimension = 6 # Magic number, 3D + Axis Angle
        self.action_tensor = tf.placeholder(tf.float32, shape=[batch_size, 1, self.action_space_dimension], name='CActionPh')
        self.rgb_tensor = tf.placeholder(tf.float32, shape=common_shape+[3], name='RgbPh')
        self.dep_tensor = tf.placeholder(tf.float32, shape=common_shape+[1], name='DepPh')

        assert self.view_num == 1 or args.sharedmultiview, "must be shared multiview, or single view"
        assert args.ferev == 13, 'Assumes --ferev 13 (implied by --visionformula 4)'
        # Let's try ResNet 18 True
        self.feature_extractor = vision.FeatureExtractorResNet(
                config.SV_RESNET18_TRUE,
                args.fehidden + [args.featnum], 'VisionNetRev13', args.elu,
                batch_normalization=batch_normalization)
        self._vision_params, self.featvec = self.feature_extractor.infer(
                self.rgb_tensor, self.dep_tensor)
        B,V,N = self.featvec.shape
        self.joint_featvec = tf.reshape(self.featvec, [-1, 1, int(V)*int(N)])

        naming = 'TunnelFinderNet'
        self._finder_net = vision.ConvApplier(None, args.polhidden, naming, args.elu)
        self._finder_params, self.finder_pred = self._finder_net.infer(fv)

class TunnelFinderTrainer(object):

    def __init__(self,
                 advcore,
                 args,
                 learning_rate,
                 batch_normalization=None):
        assert args.samplein, '--train tunnel_finder needs --samplein to indicate the tunnel vertices'
        self.tunnel_v = np.load(args.samplein)['TUNNELV']
        self.advcore = advcore
        self.args = args
        global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step

        self.loss = tf.nn.mean_squared_error(advcore.action_tensor, advcore.finder_pred)
        tf.summary.scalar('finder_loss', self.loss)

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

    def _sample_one(self, envir):
        q = uw_random.random_state(0.75)
        envir.qstate = q
        vstate = envir.vstate
        distances = pyosr.multi_distance(q, self.tunnel_v)
        ni = np.argmin(distances)
        tr,aa,dq = pyosr.differential(q, close)
        return [vstate, np.concatenate([ctr, caa], axis=-1)]

    def train(self, envir, sess, tid=None, tmax=-1):
        samples = [self._sample_one(envir) for i in range(args.batch)]
        batch_rgb = [s[0][0] for s in samples]
        batch_dep = [s[0][1] for s in samples]
        batch_dq = [s[1] for s in samples]
        ndic = {
                'rgb' : batch_rgb,
                'dep' : batch_dep,
                'dq' : batch_dq
               }
        self.dispatch_training(sess, ndic)

    def dispatch_training(self, sess, ndic, debug_output=True):
        advcore = self.advcore
        dic = {
                advcore.rgb: ndic['rgb'],
                advcore.dep: ndic['dep'],
                self.action_tensor : ndic['locomo'],
              }
        if self.summary_op is not None:
            self._log("running training op")
            _, summary, gs = sess.run([self.train_op, self.summary_op, self.global_step], feed_dict=dic)
            self._log("training op for gs {} finished".format(gs))
            # grads = curiosity.sess_no_hook(sess, self._debug_grad_op, feed_dict=dic)
            # print("[DEBUG] grads: {}".format(grads))
            self.train_writer.add_summary(summary, gs)
        else:
            sess.run(self.train_op, feed_dict=dic)

def create_advcore(learning_rate, args, batch_normalization):
    if args.train == 'tunnel_finder':
        return TunnelFinderCore(learning_rate=learning_rate, args=args, batch_normalization=batch_normalization)
    assert False, "tunnel.create_advcore: unimplemented advcore factory for --train {}".format(args.train)
