import tensorflow as tf
import numpy as np
from operator import mul

class VisionLayerConfig:
    strides = [1, 2, 2, 1]
    kernel_size = [3, 3]
    ch_in = -1                      # -1 means all channels from previous layer
    ch_out = None                   # Undefined by default
    padding = 'SAME'
    weight = None
    bias = None
    naming = None
    elu = False

    def __init__(self, ch_out, strides=[1, 2, 2, 1], kernel_size=[3,3], padding = 'SAME'):
        self.ch_out = ch_out
        self.strides = strides
        self.kernel_size = kernel_size
        self.padding = padding

    def create_kernel(self, prev_ch_out):
        if self.weight is not None:
            return self.weight, self.bias
        ch_in = self.ch_in if self.ch_in > 0 else int(prev_ch_out)
        ch_out = self.ch_out
        [w, h] = self.kernel_size[0:2]
        d = 1.0 / np.sqrt(ch_in * w * h)
        weight_shape = [1, w, h, ch_in, ch_out]
        bias_shape = [self.ch_out]
        # print('weight_shape {}'.format(weight_shape))
        # print('d {}'.format(d))
        if self.naming is None:
            weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
            bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        else:
            with tf.variable_scope(self.naming):
                weight = tf.get_variable("weight", weight_shape, tf.float32,
                        tf.random_uniform_initializer(-d, d))
                bias_name = "bias_elu" if self.elu else "bias"
                bias   = tf.get_variable(bias_name, bias_shape, tf.float32,
                        tf.random_uniform_initializer(-d, d))
        self.weight = weight
        self.bias = bias
        return weight, bias

    def apply_layer(self, prev_layer):
        print('prev {}'.format(prev_layer.shape))
        prev_ch_out = int(prev_layer.shape[-1])
        w,b = self.create_kernel(prev_ch_out)
        strides = [1] + self.strides
        conv = tf.nn.conv3d(prev_layer, w, strides, self.padding)
        # out = tf.nn.relu(conv + b)
        out = tf.nn.elu(conv + b) if self.elu else tf.nn.relu(conv + b)
        print('after CNN {}'.format(out.shape))
        return w,b,out

    @staticmethod
    def createFromDict(visdict):
        viscfg_array = []
        for visdesc in visdict:
            print(visdesc)
            viscfg = VisionLayerConfig(visdesc['ch_out'])
            if visdesc['strides'] is not None:
                viscfg.strides = visdesc['strides']
            if visdesc['kernel_size'] is not None:
                viscfg.kernel_size = visdesc['kernel_size']
            viscfg_array.append(viscfg)
        return viscfg_array

class FCLayerConfig:
    ch_in = -1
    ch_out = None
    weight = None
    bias = None
    naming = None
    elu = False

    def __init__(self, ch_out):
        self.ch_out = ch_out

    def create_kernel(self, prev_shape):
        if self.weight is not None:
            return self.weight, self.bias
        ch_in = self.ch_in if self.ch_in > 0 else int(prev_shape)
        ch_out = self.ch_out
        weight_shape = [ch_in, ch_out]
        bias_shape = [ch_out]
        d = 1.0 / np.sqrt(ch_in)
        if self.naming is None:
            weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
            bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        else:
            with tf.variable_scope(self.naming):
                weight = tf.get_variable("weight", weight_shape, tf.float32,
                        tf.random_uniform_initializer(-d, d))
                bias_name = "bias_elu" if self.elu else "bias"
                bias   = tf.get_variable(bias_name, bias_shape, tf.float32,
                        tf.random_uniform_initializer(-d, d))
        self.weight, self.bias = weight, bias
        return weight, bias

    # FIXME: ACCEPT 2D TENSOR [None, N]
    def apply_layer(self, prev):
        # Dim 0 is BATCH
        print('prev {}'.format(prev.shape))
        if len(prev.shape.as_list()) > 2:
            flatten = tf.reshape(prev, [-1, prev.shape[1].value, reduce(mul, prev.shape[2:].as_list(), 1)])
        else:
            flatten = prev
        print('flatten {}'.format(flatten.shape))
        w,b = self.create_kernel(flatten.shape[-1])
        print('w,b {} {}'.format(w.get_shape(), b.get_shape()))
        td = tf.tensordot(flatten, w, [[2], [0]])
        print('tdshape {}'.format(td.get_shape()))
        td.set_shape([None, prev.shape[1].value, self.ch_out])
        print('tdshape set to {}'.format(td.get_shape()))
        out = tf.nn.elu(td + b) if self.elu else tf.nn.relu(td + b)
        return w,b,out

class VisionNetwork:
    '''
    VisionNetwork: apply CNN+Single FC over [BVWHC] input to get [BVF] output

                   F is specified by feature_number, if feature_number is -1, F
                   is simply flattened from the last CNN layer
    '''
    thread_index = 0
    layer_number = 0
    input_tensor = None
    layer_configs = None
    view_number = 0
    feature_number = -1
    nn_layers = None # 0 is input, -1 is output
    nn_args = None
    nn_filters = None
    nn_featvec = None
    naming = None

    def __init__(self,
            input_shape,
            layer_configs,
            thread_index,
            feature_number = -1,
            input_tensor = None,
            naming = None):
        nn_layers = [] # 0 is input, -1 is output
        nn_args = []
        nn_filters = []
        net = "vis_tidx_{}".format(thread_index)
        self.thread_index = thread_index
        self.layer_configs = layer_configs
        self.feature_number = feature_number
        if input_tensor is None:
            self.input_tensor = tf.placeholder(tf.float32, shape=input_shape)
        else:
            self.input_tensor = input_tensor
            input_shape = input_tensor.shape
        self.view_number = int(input_shape[1])
        self.naming = naming

    def get_output_tensors(self):
        if not self.nn_layers:
            self.infer()
        return self.nn_featvec

    features = property(get_output_tensors)

    def infer(self, alternative_input_tensor=None):
        if self.naming is None:
            infer_impl(self, alternative_input_tensor)
        else:
            with tf.variable_scope(self.naming):
                infer_impl(self, alternative_input_tensor)

    def infer_impl(self, alternative_input_tensor):
        self.nn_layers = [self.input_tensor]
        self.nn_args = []
        for conf in self.layer_configs:
            prev_layer = self.nn_layers[-1]
            print("prev shape {}".format(prev_layer.shape))
            prev_ch = prev_layer.shape[-1]
            #w,b = conf.create_kernel(prev_ch)
            #conv = tf.nn.conv2d(prev_layer, w, conf.strides, conf.padding)
            #out = tf.nn.relu(conv + b)
            w,b,out = conf.apply_layer(prev_layer)
            self.nn_layers.append(out)
            self.nn_args.append((w,b))
        if self.feature_number < 0:
            shape_BV = [int(self.input_shape[0]), int(self.input_shape[1]), -1]
            self.nn_featvec = tf.reshape(self.nn_layers[-1], shape_BV)
            return
        fclc = FCLayerConfig(self.feature_number)
        # self.mv_cnnflatten = tf.reshape(self.nn_layers[-1], [self.view_number, -1])
        # print('mv_cnn flatten: {}'.format(self.mv_cnnflatten.shape))
        #w,b = fclc.create_kernel(self.mv_cnnflatten.shape[-1])
        #self.mv_featvec = tf.nn.relu(tf.matmul(self.mv_cnnflatten, w) + b)

        # FLATTEN is handled by FCLayerConfig
        w,b,self.mv_featvec = fclc.apply_layer(self.nn_layers[-1])
        print("mv_featvec {}".format(self.mv_featvec.get_shape()))
        self.nn_args.append((w,b))
        self.nn_featvec = self.mv_featvec

    def get_nn_args(self):
        '''
        Return a flatten list from the stored list of tuples.
        '''
        return list(sum(self.nn_args, ()))


def create_view_array_from_config(view_config):
    view_array = []
    for angle,ncam in view_config:
        view_array += [ [angle,float(i)] for i in np.arange(0.0, 360.0, 360.0/float(ncam)) ]
    return view_array

def ExtractPerViewFeatures(input_shape,
        input_tensor,
        confdict,
        viewnum,
        featnum_sqroot,
        post_process=True):
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
    '''
    featnum = featnum_sqroot ** 2
    vision_net = VisionNetwork(input_shape,
            VisionLayerConfig.createFromDict(confdict),
            0, # TODO: multi-threading
            featnum,
            input_tensor)
    if not post_process:
        return vision_net, vision_net.features
    # print('sv_net.featvec.shape = {}'.format(sv_net.features.shape))

    # Reshape to [B,V,f,f,1], where F = f*f
    # So we can apply CNN to [f,f,V] images by treating V as channels.
    sq_featvec = tf.reshape(vision_net.features, [-1, viewnum, featnum_sqroot, featnum_sqroot, 1])
    # print('sq_svfeatvec.shape = {}'.format(sq_svfeatvec.shape))
    # Transpose BVff1 to B1ffV
    featvec = tf.transpose(sq_featvec, [0,4,2,3,1])
    return vision_net, featvec


def ExtractAllViewFeatures(
        rgb_tensor,
        depth_tensor,
        view_config,
        svconfdict,
        mvconfdict,
        intermediate_featnum_sqroot,
        final_featnum):
    w = int(rgb.tensor.get_shape()[2])
    h = int(rgb.tensor.get_shape()[3])
    view_num = len(view_array)
    sv_rgb_net, sv_rgb_featvec = ExtractPerViewFeatures(rgb_tensor, None,
            view_num, intermediate_featnum_sqroot)
    sv_depth_net, sv_depth_featvec = ExtractPerViewFeatures(depth_tensor, None,
            view_num, intermediate_featnum_sqroot)
    sv_featvec = tf.concat([sv_depth_featvec, sv_rgb_featvec], 4)
    mv_net, mv_featvec = ExtractPerViewFeatures(sv_featvec, None, 1,
            featnum_sqroot, post_process=False)
    params = []
    params.append(sv_depth_net.get_nn_args())
    params.append(sv_rgb_net.get_nn_args())
    params.append(mv_net.get_nn_args())
    return sum(params,[]), mv_featvec

class ConvApplier:
    layer_configs = None
    naming = None
    elu = False

    '''
    Input: [B, V, W, H, C]
        - Batch, View, Width, Height, Channel
    Output: [B, V, F]
        - F: featnums
    '''

    def __init__(self,
            confdict,
            featnums,
            naming=None,
            elu=False):
        self.layer_configs = []
        self.elu = elu
        if confdict is not None:
            self.layer_configs = VisionLayerConfig.createFromDict(confdict)
        for featnum in featnums:
            self.layer_configs.append(FCLayerConfig(featnum))
        self.naming = naming
        # print("== Init Len of layer_configs {}, confdict {}, featnums {}".format(len(self.layer_configs), confdict, featnums))

    def infer(self, input_tensor):
        if self.naming is None:
            return self.infer_impl(input_tensor)
        else:
            with tf.variable_scope(self.naming):
                return self.infer_impl(input_tensor)

    def infer_impl(self, input_tensor):
        for layer in self.layer_configs:
            layer.elu = self.elu
        nn_layer_tensor = [input_tensor]
        nn_args = []
        index = 0
        # print("== Len of layer_configs {}".format(len(self.layer_configs)))
        for conf in self.layer_configs:
            if self.naming is not None:
                # print("== Index {}".format(index))
                conf.naming = "Layer_{}".format(index)
                index += 1
            prev_layer_tensor = nn_layer_tensor[-1]
            prev_ch = prev_layer_tensor[-1]
            w,b,out = conf.apply_layer(prev_layer_tensor)
            nn_layer_tensor.append(out)
            nn_args.append((w,b))
        return nn_args, nn_layer_tensor[-1]

def GetFeatureSquare(featvec):
    shape = featvec.shape.as_list()
    view_num = int(shape[1])
    featnum = reduce(mul, shape[2:], 1)
    featnum_sqroot = int(np.sqrt(featnum))
    sq_featvec = tf.reshape(featvec, [-1, view_num, featnum_sqroot, featnum_sqroot, 1])
    return tf.transpose(sq_featvec, [0,4,2,3,1])

def GetCombineFeatureSquare(rgb_featvec, depth_featvec):
    rgb_featsq = GetFeatureSquare(rgb_featvec)
    depth_featsq = GetFeatureSquare(depth_featvec)
    combine_featsq = tf.concat([rgb_featsq, depth_featsq], 4)
    return combine_featsq


'''
Basic Feature Extractor
Arch:
    CNN := (FC * CNN^k)
    SVFE := cat(CNN(rgb), CNN(dep))
    MVIN := reshape(shape=(m,m), cat(SVFE_0, SVFE_1, ... , SVFE_k-1))
    FE := CNN(MVIN)
    output = FE(rgb, dep)
'''
class FeatureExtractor:
    view_array = None
    rgb_conv_applier = None
    depth_conv_applier = None
    combine_conv_applier = None
    naming = None
    reuse = False

    def __init__(self,
            svconfdict,
            mvconfdict,
            intermediate_featnum,
            final_featnum,
            naming,
            elu):
        self.rgb_conv_applier = ConvApplier(svconfdict, [intermediate_featnum], 'PerViewRGB', elu)
        self.depth_conv_applier = ConvApplier(svconfdict, [intermediate_featnum], 'PerViewDepth', elu)
        # self.mv_shape = [None, 1, intermediate_featnum_sqroot, intermediate_featnum_sqroot, view_num]
        self.combine_conv_applier = ConvApplier(mvconfdict, [final_featnum], 'CombineViewRGBD', elu)
        self.naming = naming

    def infer(self, rgb_input, depth_input):
        with tf.variable_scope(self.naming, reuse=self.reuse) as scope:
            rgb_nn_params, rgb_nn_featvec = self.rgb_conv_applier.infer(rgb_input)
            depth_nn_params, depth_nn_featvec = self.depth_conv_applier.infer(depth_input)
            combine_featsq = GetCombineFeatureSquare(rgb_nn_featvec, depth_nn_featvec)
            combine_params, combine_out = self.combine_conv_applier.infer(combine_featsq)
        self.reuse = True
        return rgb_nn_params + depth_nn_params + combine_params, combine_out


'''
Feature Extractor Rev. 2
Arch:
    CNN := (FC^n * CNN^m)
    SVFE(rgb,dep) := cat(CNN(rgb), CNN(dep))
    FE := FC^K * SVFE
    output = FE(rgb, dep)
'''
class FeatureExtractorRev2:
    naming = None
    reuse = None

    def __init__(self,
            svconfdict,
            intermediate_featnum,
            final_featnums,
            naming,
            elu):
        '''
        Note: we have two FCs rather than one
        '''
        self.rgb_conv_applier = ConvApplier(svconfdict, [intermediate_featnum, intermediate_featnum], 'PerViewRGB', elu)
        self.depth_conv_applier = ConvApplier(svconfdict, [intermediate_featnum, intermediate_featnum], 'PerViewDepth', elu)
        self.mv_fc_applier = ConvApplier(None, final_featnums, "CombineViewRGBDFC", elu)

        self.naming = naming

    def infer(self, rgb_input, depth_input):
        with tf.variable_scope(self.naming, reuse=self.reuse) as scope:
            rgb_nn_params, rgb_nn_featvec = self.rgb_conv_applier.infer(rgb_input)
            depth_nn_params, depth_nn_featvec = self.depth_conv_applier.infer(depth_input)
            combine_in = tf.concat([rgb_nn_featvec, depth_featvec], axis=-1)
            combine_params, combine_out = self.mv_fc_applier.infer(combine_in)
        self.reuse = True
        return rgb_nn_params + depth_nn_params + combine_params, combine_out


'''
Feature Extractor Rev. 3
Arch:
    CNN := (FC^n * CNN^m)
    SVFE(rgb,dep) := CNN(cat(rgb,dep))
    FE := FC^K * SVFE
    output = FE(rgb, dep)
'''
class FeatureExtractorRev3:
    naming = None
    reuse = None

    def __init__(self,
            svconfdict,
            intermediate_featnum,
            final_featnums,
            naming,
            elu):
        '''
        Note: we have two FCs rather than one
        '''
        self.rgbd_conv_applier = ConvApplier(svconfdict, [intermediate_featnum, intermediate_featnum], 'PerViewRGBD', elu)
        '''
        RGBDCombineViewFC:
            We put RGBD at the beginning because we combine RGB and D firstly.
        '''
        self.mv_fc_applier = ConvApplier(None, final_featnums, "RGBDCombineViewFC", elu)

        self.naming = naming

    def infer(self, rgb_input, depth_input):
        with tf.variable_scope(self.naming, reuse=self.reuse) as scope:
            rgbd_input = tf.concat([rgb_input, depth_input], axis=-1)
            rgbd_nn_params, rgbd_nn_featvec = self.rgbd_conv_applier.infer(rgbd_input)
            combine_params, combine_out = self.mv_fc_applier.infer(rgbd_nn_params)
        self.reuse = True
        return rgbd_nn_params + combine_params, combine_out
