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
        w = h = args.res
        self.args = args
        self.batch_normalization = batch_normalization
        batch_size = None if args.EXPLICIT_BATCH_SIZE < 0 else args.EXPLICIT_BATCH_SIZE
        self.batch_size = batch_size

        common_shape = [batch_size, self.view_num, w, h]
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
        self._finder_net = vision.ConvApplier(None, args.polhidden + [self.action_space_dimension], naming, args.elu)
        self._finder_params, self.finder_pred = self._finder_net.infer(self.joint_featvec)

    '''
    Stub property
    '''
    @property
    def softmax_policy(self):
        return 0

class TunnelFinderTrainer(object):

    def __init__(self,
                 advcore,
                 args,
                 learning_rate,
                 batch_normalization=None):
        assert args.samplein, '--train tunnel_finder needs --samplein to indicate the tunnel vertices'
        self._tunnel_v = np.load(args.samplein)['TUNNEL_V']
        self.unit_tunnel_v = None
        self.advcore = advcore
        self.args = args
        global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.loss = tf.losses.mean_squared_error(advcore.action_tensor, advcore.finder_pred)
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

        self.sample_index = 0

    @property
    def total_iter(self):
        return self.args.iter

    def _sample_one(self, envir):
        q = uw_random.random_state(0.75)
        if self.unit_tunnel_v is None:
            self.unit_tunnel_v = np.array([envir.r.translate_to_unit_state(v) for v in self._tunnel_v])
        envir.qstate = q
        vstate = envir.vstate
        distances = pyosr.multi_distance(q, self.unit_tunnel_v)
        ni = np.argmin(distances)
        close = self.unit_tunnel_v[ni]
        tr,aa,dq = pyosr.differential(q, close)
        assert pyosr.distance(close, pyosr.apply(q, tr, aa)) < 1e-6, "pyosr.differential is not pyosr.apply ^ -1"
        return [vstate, np.concatenate([tr, aa], axis=-1), q, close]

    def train(self, envir, sess, tid=None, tmax=-1):
        samples = [self._sample_one(envir) for i in range(self.args.batch)]

        '''
        --sampleout support
        '''
        if self.args.sampleout:
            fn = '{}/{}.npz'.format(self.args.sampleout, self.sample_index)
            qs = np.array([s[2] for s in samples])
            dqs = np.array([s[1] for s in samples])
            closes = np.array([s[3] for s in samples])
            np.savez(fn, QS=qs, DQS=dqs, CLOSES=closes)
        self.sample_index += 1

        batch_rgb = [s[0][0] for s in samples]
        batch_dep = [s[0][1] for s in samples]
        batch_dq = [[s[1]] for s in samples]
        ndic = {
                'rgb' : batch_rgb,
                'dep' : batch_dep,
                'dq' : batch_dq
               }
        self.dispatch_training(sess, ndic)

    def _log(self, text):
        print(text)

    def dispatch_training(self, sess, ndic, debug_output=True):
        advcore = self.advcore
        dic = {
                advcore.rgb_tensor: ndic['rgb'],
                advcore.dep_tensor: ndic['dep'],
                advcore.action_tensor : ndic['dq'],
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
