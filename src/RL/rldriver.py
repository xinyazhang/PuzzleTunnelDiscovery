import pyosr
import numpy as np
import tensorflow as tf
import vision

class RLDriver:
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
            input_tensor):
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
        colorshape = [len(view_array), w, h, CHANNEL]
        sv_color = vision.VisionNetwork(colorshape,
                vision.VisionLayerConfig.createFromDict(svconfdict),
                0, # FIXME: multi-threading
                sv_sqfeatnum ** 2)
        sq_svfeatvec = tf.reshape(sv_color.features, [-1, sv_sqfeatnum, sv_sqfeatnum, 1])
        print('sv_color.featvec.shape = {}'.format(sv_color.features.shape))
        sv_featvec = tf.transpose(sq_svfeatvec, [3,1,2,0])

        mv_color = vision.VisionNetwork(None,
                vision.VisionLayerConfig.createFromDict(svconfdict),
                0, # FIXME: multi-threading
                mv_featnum,
                sv_featvec)
        self.sv_colorfv = sv_color.features
        self.mv_colorfv = mv_color.features
        conf_final_fc = vision.FCLayerConfig(output_number)
        w,b,final = conf_final_fc.apply_layer(mv_color.features)
        self.final = final

