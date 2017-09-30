import pyosr
import numpy as np
import tensorflow as tf
import vision

class RLDriver:
    '''
    RLDriver: driver the RL process from given configurations including:
        - Robot models and states - NN configuration '''
    renderer = None
    master = None
    sv_rgb_net = None
    sv_depth_net = None
    mv_net = None
    decision_net_args = []
    sync_op_group = None

    '''
        models: files name of [model, robot], or [model], or [model, None]
        view_config: array of (angle, number cameras)
        svconfdict: network configuration dict for Single View (SV) CNN, e.g. config.SV_VISCFG
        mvconfdict: network configuration dict for Multiview View (MV) CNN
        sv_sqfeatnum: Squared Root of feature numbers for SV CNN
    '''
    def __init__(self,
            models,
            init_state,
            view_config,
            svconfdict,
            mvconfdict,
            output_number = 3 * 2 * 2, # For RL: X,Y,Z * (rotate,translate) * (pos,neg)
            sv_sqfeatnum = 16,
            mv_featnum = 256,
            input_tensor = None,
            use_rgb = False,
            master_driver = None):
        self.renderer = pyosr.Renderer()
        r = self.renderer
        if master_driver is None:
            r.setup()
            r.loadModelFromFile(models[0])
            if len(models) > 1 and models[1] is not None:
                r.loadRobotFromFile(models[1])
                r.state = np.array(init_state, dtype=np.float32)
            print('robot loaded')
            r.scaleToUnit()
            r.angleModel(0.0, 0.0)
            r.default_depth = 0.0
        else:
            r.setupFrom(master_driver.renderer)
            r.default_depth = master_driver.renderer.default_depth
        self.master = master_driver

        view_array = []
        for angle,ncam in view_config:
            view_array += [ [angle,float(i)] for i in np.arange(0.0, 360.0, 360.0/float(ncam)) ]
        r.views = np.array(view_array, dtype=np.float32)

        w = r.pbufferWidth
        h = r.pbufferHeight
        # TODO: Switch to RGB-D rather than D-only
        CHANNEL = 1
        colorshape = [1, len(view_array), w, h, CHANNEL] if input_tensor is None else None
        self.sv_depth_net, sv_depth_featvec = self._create_sv_features(colorshape,
                input_tensor,
                svconfdict,
                len(view_array),
                sv_sqfeatnum)
        print('sv_depth_featvec: {}'.format(sv_depth_featvec.shape))
        if use_rgb is True:
            CHANNEL = 3
            # input_tensor is always for depth
            colorshape = [1, len(view_array), w, h, CHANNEL]
            self.sv_rgb_net, sv_rgb_featvec = self._create_sv_features(colorshape,
                    None,
                    svconfdict,
                    len(view_array),
                    sv_sqfeatnum)
            # Concat B1WHV and B1WHV into B1WHV
            print('sv_rgb_featvec: {}'.format(sv_rgb_featvec.shape))
            sv_featvec = tf.concat([sv_depth_featvec, sv_rgb_featvec], 4)
        else:
            self.sv_rgb_net = None
            sv_rgb_featvec = None
            sv_featvec = sv_depth_featvec
        print('sv_featvec: {}'.format(sv_featvec.shape))

        mv_net = vision.VisionNetwork(None,
                vision.VisionLayerConfig.createFromDict(mvconfdict),
                0, # FIXME: multi-threading
                mv_featnum,
                sv_featvec)
        self.mv_net = mv_net
        self.sv_depthfv = sv_depth_featvec
        self.sv_rgbfv = sv_rgb_featvec
        self.mv_fv = mv_net.features
        conf_final_fc = vision.FCLayerConfig(output_number)
        w,b,final = conf_final_fc.apply_layer(mv_net.features)
        final = tf.contrib.layers.flatten(final)
        print('rldriver output {}'.format(final.shape))
        self.final = final
        self.decision_net_args = [w,b]

    def _create_sv_features(self, input_shape, input_tensor, svconfdict, num_views, sv_sqfeatnum):
        '''
        Create Per-View NN from given configuration

            input_shape: list to specify [B, V, W, H, C]
                * Batch, View, Width, Height, Channel
            input_tensor: tensor as the input of this NN
                * Shape must be [B, V, W, H, C]
                * only one of input_shape and input_tensor is required
            svconfdict: NN configuration dict, see config.SV_VISCFG for example

            RETURN
            Per-View Feature Vector in [B, 1, W, H, V]

            TODO: self.thread_id
        '''
        sv_net = vision.VisionNetwork(input_shape,
                vision.VisionLayerConfig.createFromDict(svconfdict),
                0, # TODO: multi-threading
                sv_sqfeatnum ** 2,
                input_tensor)
        print('sv_net.featvec.shape = {}'.format(sv_net.features.shape))
        # Reshape to [B,V,f,f,1], where F = f*f
        # So we can apply CNN to [f,f,V] images by treating V as channels.
        sq_svfeatvec = tf.reshape(sv_net.features, [-1, num_views, sv_sqfeatnum, sv_sqfeatnum, 1])
        print('sq_svfeatvec.shape = {}'.format(sq_svfeatvec.shape))
        # Transpose BVff1 to B1ffV
        sv_featvec = tf.transpose(sq_svfeatvec, [0,4,2,3,1])
        return sv_net, sv_featvec

    def get_nn_args(self):
        args = []
        args.append(self.sv_depth_net.get_nn_args())
        if self.sv_rgb_net is not None:
            args.append(self.sv_rgb_net.get_nn_args())
        args.append(self.mv_net.get_nn_args())
        return sum(args, []) # Concatenate (+) all element in args, which is a list of list.

    def get_sync_from_master_op(self):
        if self.sync_op_group is not None:
            return self.sync_op_group
        master_args = self.master.get_nn_args()
        self_args = self.get_nn_args()
        sync_ops = []

        for src,dst in zip(master_args, self_args):
            sync_op = tf.assign(dst, src)
            sync_ops.append(sync_op)

        self.sync_op_group = tf.group(*sync_ops)
        return self.sync_op_group

    def get_reward(self, action):
        nstate, done, ratio = r.transit_state(r.state, action,
                config.MAGNITUDES, config.STATE_CHECK_DELTAS)
        reward = 0.0
        if not done and ratio > 0.0:
            reward = 1.0
        if numpy.linalg.norm(nstate[0:3]) > 1.0:
            reward = 64.0
        return nstate, reward
