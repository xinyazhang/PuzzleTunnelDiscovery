from __future__ import print_function
import numpy as np
import tensorflow as tf
import pyosr
import rlutil
import tfutil
import vision
import config
import curiosity
import uw_random

'''
TunnelFinderCore:
    Class similar to IAdvantageCore for tunnel_finder approaches.

    Note: we do not need full IAdvantageCore instance since tunnel_finder approaches are not RL.
'''
class TunnelFinderCore(object):

    def __init__(self, learning_rate, args, batch_normalization=None):
        self._debug_trans_only = 'tunnel_finder_trans_only' in args.debug_flags

        self.view_num, self.views = rlutil.get_view_cfg(args)
        w = h = args.res
        self.args = args
        self.batch_normalization = batch_normalization
        batch_size = None if args.EXPLICIT_BATCH_SIZE < 0 else args.EXPLICIT_BATCH_SIZE
        self.batch_size = batch_size
        # tf.reshape does not accept None as dimension
        self.reshape_batch_size = -1 if batch_size is None else batch_size

        common_shape = [batch_size, self.view_num, w, h]
        self.action_space_dimension = 6 # Magic number, 3D + Axis Angle
        self.pred_action_size = self.action_space_dimension * self.get_number_of_predictions()
        action_shape = self._get_action_placeholder_shape(batch_size)
        self.action_tensor = tf.placeholder(tf.float32, shape=action_shape, name='CActionPh')
        self.rgb_tensor = tf.placeholder(tf.float32, shape=common_shape+[3], name='RgbPh')
        self.dep_tensor = tf.placeholder(tf.float32, shape=common_shape+[1], name='DepPh')

        assert self.view_num == 1 or args.sharedmultiview, "must be shared multiview, or single view"
        assert args.ferev in [11, 13], 'Assumes --ferev 11 or 13 (implied by --visionformula )'
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
        self._finder_net = vision.ConvApplier(None,
                args.polhidden + [self.pred_action_size],
                naming, args.elu)
        self._finder_params, self.finder_pred = self._finder_net.infer(self.joint_featvec)

    def _get_action_placeholder_shape(self, batch_size):
        return [batch_size, 1, self.action_space_dimension]

    def get_number_of_predictions(self):
        return 1

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
                 batch_normalization=None,
                 build_summary_op=True):
        self._debug_trans_only = 'tunnel_finder_trans_only' in args.debug_flags

        assert args.samplein, '--train tunnel_finder needs --samplein to indicate the tunnel vertices'
        self._tunnel_v = np.load(args.samplein)['TUNNEL_V'][:args.sampletouse]
        self.unit_tunnel_v = None
        self.advcore = advcore
        self.args = args
        global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self._build_loss(advcore)

        tf.summary.scalar('finder_loss', self.loss)

        if batch_normalization is not None:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
        else:
            self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
        if build_summary_op:
            self.build_summary_op()

        self.sample_index = 0

    def _build_loss(self, advcore):
        if 'se3_geodesic_loss' not in self.args.debug_flags:
            self.loss = tf.losses.mean_squared_error(advcore.action_tensor, advcore.finder_pred)
        else:
            e3_distancce = tf.nn.l2_loss(advcore.action_tensor[:,:,:3] -
                                         advcore.finder_pred[:,:,:3])
            e3_distancce = tf.reduce_sum(e3_distancce)
            if self.action_input_size > 3:
                o3_distance = tfutil.axis_angle_geodesic_distance(advcore.action_tensor[:,:,3:],
                                                                  advcore.finder_pred[:,:,3:],
                                                                  keepdims=True)
                o3_distance = tf.reduce_sum(o3_distance)
                self.loss = e3_distancce + o3_distance
            else:
                self.loss = e3_distancce

    def build_summary_op(self):
        ckpt_dir = self.args.ckptdir
        if ckpt_dir is not None:
            self.summary_op = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(ckpt_dir + '/summary', tf.get_default_graph())
        else:
            self.summary_op = None
            self.train_writer = None

    @property
    def total_iter(self):
        return self.args.iter

    def _get_gt_action_from_sample(self, q):
        distances = pyosr.multi_distance(q, self.unit_tunnel_v)
        ni = np.argmin(distances)
        close = self.unit_tunnel_v[ni]
        tr,aa,dq = pyosr.differential(q, close)
        assert pyosr.distance(close, pyosr.apply(q, tr, aa)) < 1e-6, "pyosr.differential is not pyosr.apply ^ -1"
        return np.concatenate([tr, aa], axis=-1), close

    def _sample_one(self, envir, istate=None):
        if istate is None:
            '''
            if self._debug_trans_only:
                q = 0.75 * (np.random.rand(3) - 0.5)
                q = np.array([q[0],q[1],q[2],1.0,0.0,0.0,0.0], dtype=np.float32)
            else:
            '''
            q = uw_random.random_state(0.75)
        else:
            q = istate
        if self.unit_tunnel_v is None:
            self.unit_tunnel_v = np.array([envir.r.translate_to_unit_state(v) for v in self._tunnel_v])
        envir.qstate = q
        if self._debug_trans_only:
            # Enable perturbation
            manual_p = uw_random.random_state(0.5)
            envir.r.set_perturbation(manual_p)
        vstate = envir.vstate
        tr_and_aa, close = self._get_gt_action_from_sample(q)
        if self._debug_trans_only:
            # Accumulate translation
            tr_and_aa[:3] += manual_p[:3]
            q[:3] += manual_p[:3]
        return [vstate, tr_and_aa, q, close]

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

class TunnelFinderTwin1(TunnelFinderCore):

    def __init__(self, learning_rate, args, batch_normalization=None):
        super(TunnelFinderTwin1, self).__init__(learning_rate=learning_rate,
                args=args, batch_normalization=batch_normalization)

        self.coarse_action = self.action_tensor
        self.coarse_rgb = self.rgb_tensor
        self.coarse_dep = self.dep_tensor
        self.fine_action = self.action_tensor
        self.fine_rgb = self.rgb_tensor
        self.fine_dep = self.dep_tensor

        self._coarse_net = self._finder_net
        self.coarse_pred = self.finder_pred
        naming = 'TunnelFinderNet_Fine'
        self._fine_net = vision.ConvApplier(None, args.polhidden + [self.action_space_dimension], naming, args.elu)
        self._fine_params, self.fine_pred = self._fine_net.infer(self.joint_featvec)

    '''
    Stub property
    '''
    @property
    def softmax_policy(self):
        return 0

class TunnelFinderTwinTrainer(TunnelFinderTrainer):

    def __init__(self,
                 advcore,
                 args,
                 learning_rate,
                 batch_normalization=None):
        super(TunnelFinderTwinTrainer, self).__init__(
                advcore=advcore,
                args=args,
                learning_rate=learning_rate,
                batch_normalization=batch_normalization,
                build_summary_op=False) # Do not build summary in super class

        '''
        BAG objects: guiding object from np.array to tensor

            We are training multiple networks, and this class is introduced to
        avoid duplicated code in training different networks (esp. for distrubited TF)
        '''
        self.coarse_bag = {
                'rgb': advcore.coarse_rgb,
                'dep': advcore.coarse_dep,
                'dq': advcore.coarse_action,
                }
        self.fine_bag = {
                'rgb': advcore.fine_rgb,
                'dep': advcore.fine_dep,
                'dq': advcore.fine_action,
                }
        self.coarse_loss = self.loss
        self.fine_loss = tf.losses.mean_squared_error(advcore.fine_action, advcore.fine_pred)
        tf.summary.scalar('coarse_loss', self.coarse_loss)
        tf.summary.scalar('fine_loss', self.fine_loss)
        self.coarse_train_op = self.train_op
        assert batch_normalization is None
        self.fine_train_op = self.optimizer.minimize(self.fine_loss, global_step=self.global_step)
        self.build_summary_op() # build summary op here

    def ndic_bag_inner_join(self, ndic, bag):
        dic = {}
        for k in bag.keys():
            dic[bag[k]] = ndic[k]
        return dic

    def train(self, envir, sess, tid=None, tmax=-1):
        advcore = self.advcore
        '''
        Train coarse net
        '''
        samples = [self._sample_one(envir) for i in range(self.args.batch)]

        batch_rgb = [s[0][0] for s in samples]
        batch_dep = [s[0][1] for s in samples]
        batch_dq = [[s[1]] for s in samples]
        ndic = {
                'rgb' : batch_rgb,
                'dep' : batch_dep,
                'dq' : batch_dq
               }
        self.dispatch_training_to(sess, ndic, self.coarse_bag, self.coarse_train_op)
        dic = self.ndic_bag_inner_join(ndic, self.coarse_bag)
        [pred1] = curiosity.sess_no_hook(sess, [advcore.fine_pred], feed_dict=dic)

        '''
        Train fine net
        '''
        samples2 = [self._sample_one(envir, istate=pyosr.apply(samples[i][2], pred1[i,0,:3], pred1[i,0,3:])) for i in range(self.args.batch)]
        batch_rgb = [s[0][0] for s in samples2]
        batch_dep = [s[0][1] for s in samples2]
        batch_dq = [[s[1]] for s in samples2]
        ndic = {
                'rgb' : batch_rgb,
                'dep' : batch_dep,
                'dq' : batch_dq
               }
        self.dispatch_training_to(sess, ndic, self.fine_bag, self.fine_train_op)

    def dispatch_training_to(self, sess, ndic, tensorbag, train_op, debug_output=True):
        advcore = self.advcore
        dic = self.ndic_bag_inner_join(ndic, tensorbag)
        if self.summary_op is not None:
            self._log("running training op")
            _, summary, gs = sess.run([train_op, self.summary_op, self.global_step], feed_dict=dic)
            self._log("training op for gs {} finished".format(gs))
            # grads = curiosity.sess_no_hook(sess, self._debug_grad_op, feed_dict=dic)
            # print("[DEBUG] grads: {}".format(grads))
            self.train_writer.add_summary(summary, gs)
        else:
            sess.run(train_op, feed_dict=dic)

'''
Below are classes of ForEach1 Approach
'''
class TunnelFinderForEach1(TunnelFinderCore):

    def __init__(self, learning_rate, args, batch_normalization=None):
        assert args.samplein, '--train tunnel_finder_foreach1 needs --samplein to determine the size of prediction network'
        tunnel_v = np.load(args.samplein)['TUNNEL_V'][:args.sampletouse]
        self._pred_num = len(tunnel_v)
        super(TunnelFinderForEach1, self).__init__(learning_rate=learning_rate,
                args=args, batch_normalization=batch_normalization)
        # Helper output tensor
        self.finder_pred_foreach = tf.reshape(self.finder_pred,
                [-1, 1, self.get_number_of_predictions(), self.action_space_dimension])
        if self._debug_trans_only:
            self.finder_pred_foreach = self.finder_pred_foreach[:, :, :, :3]
        print('ForEach1: original prediction shape {}'.format(self.finder_pred.shape))
        print('ForEach1: output shape {}'.format(self.finder_pred_foreach.shape))

    def _get_action_placeholder_shape(self, batch_size):
        return [batch_size, 1, self.get_number_of_predictions(), self.action_space_dimension]

    def get_number_of_predictions(self):
        return self._pred_num

'''
TunnelFinderForEach1Trainer:
    Trainer of tunnel_finder_foreach1.
'''
class TunnelFinderForEach1Trainer(TunnelFinderTrainer):
    def __init__(self,
                 advcore,
                 args,
                 learning_rate,
                 batch_normalization=None):
        assert isinstance(advcore, TunnelFinderForEach1), 'advcore must be TunnelFinderForEach1'
        super(TunnelFinderForEach1Trainer, self).__init__(
                advcore=advcore,
                args=args,
                learning_rate=learning_rate,
                batch_normalization=batch_normalization,
                build_summary_op=True)

    def _build_loss(self, advcore):
        assert'se3_geodesic_loss' not in self.args.debug_flags, "se3_geodesic_loss is not supported by TunnelFinderForEach1Trainer"
        action_per_tunnel_v = advcore.action_tensor # No need to reshape here
        assert advcore.action_tensor.shape.as_list() == advcore.finder_pred_foreach.shape.as_list()
        self.loss = tf.losses.mean_squared_error(advcore.action_tensor, advcore.finder_pred_foreach)
        self._log('!!! action_per_tunnel_v {}'.format(action_per_tunnel_v.shape))

    def _get_gt_action_from_sample(self, q):
        dq = pyosr.multi_differential(q, self.unit_tunnel_v, with_se3=False)
        if self._debug_trans_only:
            dq = dq[:, :3]
        return dq, 0

def create_advcore(learning_rate, args, batch_normalization):
    CTOR=None
    if args.train == 'tunnel_finder':
        CTOR=TunnelFinderCore
    if args.train == 'tunnel_finder_twin1':
        CTOR=TunnelFinderTwin1
    if args.train == 'tunnel_finder_foreach1':
        CTOR=TunnelFinderForEach1
    assert CTOR is not None, "tunnel.create_advcore: unimplemented advcore factory for --train {}".format(args.train)
    return CTOR(learning_rate=learning_rate, args=args, batch_normalization=batch_normalization)
