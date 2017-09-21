import pyosr
import numpy as np
import tensorflow as tf
import vision

class RLDriver:
    '''
    RLDriver: driver the RL process from given configurations including:
        - Robot models and states
        - NN configuration
    '''
    renderer = None

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
            use_rgb = False):
        self.renderer = pyosr.Renderer()
        r = self.renderer
        r.setup()
        r.loadModelFromFile(models[0])
        if len(models) > 1 and models[1] is not None:
            r.loadRobotFromFile(models[1])
            r.state = np.array(init_state, dtype=np.float32)
        print('robot loaded')
        r.scaleToUnit()
        r.angleModel(0.0, 0.0)
        r.default_depth = 0.0
        view_array = []
        for angle,ncam in view_config:
            view_array += [ [angle,float(i)] for i in np.arange(0.0, 360.0, 360.0/float(ncam)) ]

        w = r.pbufferWidth
        h = r.pbufferHeight
        # TODO: Switch to RGB-D rather than D-only
        CHANNEL = 1
        colorshape = [1, len(view_array), w, h, CHANNEL] if input_tensor is None else None
        sv_depth_featvec = self.create_sv_net(colorshape,
                input_tensor,
                svconfdict,
                len(view_array),
                sv_sqfeatnum)
        print('sv_depth_featvec: {}'.format(sv_depth_featvec.shape))
        if use_rgb is True:
            CHANNEL = 3
            # input_tensor is always for depth
            colorshape = [1, len(view_array), w, h, CHANNEL]
            sv_rgb_featvec = self.create_sv_net(colorshape,
                    None,
                    svconfdict,
                    len(view_array),
                    sv_sqfeatnum)
            # Concat B1WHV and B1WHV into B1WHV
            print('sv_rgb_featvec: {}'.format(sv_rgb_featvec.shape))
            sv_featvec = tf.concat([sv_depth_featvec, sv_rgb_featvec], 4)
        else:
            sv_rgb_featvec = None
            sv_featvec = sv_depth_featvec
        print('sv_featvec: {}'.format(sv_featvec.shape))

        mv_net = vision.VisionNetwork(None,
                vision.VisionLayerConfig.createFromDict(svconfdict),
                0, # FIXME: multi-threading
                mv_featnum,
                sv_featvec)
        self.sv_depthfv = sv_depth_featvec
        self.sv_rgbfv = sv_rgb_featvec
        self.mv_fv = mv_net.features
        conf_final_fc = vision.FCLayerConfig(output_number)
        w,b,final = conf_final_fc.apply_layer(mv_net.features)
        final = tf.contrib.layers.flatten(final)
        print('rldriver output {}'.format(final.shape))
        self.final = final

    def create_sv_net(self, input_shape, input_tensor, svconfdict, num_views, sv_sqfeatnum):
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
        sv_color = vision.VisionNetwork(input_shape,
                vision.VisionLayerConfig.createFromDict(svconfdict),
                0, # TODO: multi-threading
                sv_sqfeatnum ** 2,
                input_tensor)
        print('sv_color.featvec.shape = {}'.format(sv_color.features.shape))
        # Reshape to [B,V,f,f,1], where F = f*f
        # So we can apply CNN to [f,f,V] images by treating V as channels.
        sq_svfeatvec = tf.reshape(sv_color.features, [-1, num_views, sv_sqfeatnum, sv_sqfeatnum, 1])
        print('sq_svfeatvec.shape = {}'.format(sq_svfeatvec.shape))
        # Transpose BVff1 to B1ffV
        sv_featvec = tf.transpose(sq_svfeatvec, [0,4,2,3,1])
        return sv_featvec
