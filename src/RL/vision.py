import tensorflow as tf
import numpy as np
from operator import mul

RESIDUAL_NONE = 0
RESIDUAL_FORK = 1
RESIDUAL_PASS = 2
RESIDUAL_JOIN = 3

GRADBIAS_ELEMENTS_PER_CHANNEL = 3

class VisionLayerConfig:
    strides = None
    kernel_size = None
    ch_in = -1                      # -1 means all channels from previous layer
    ch_out = None                   # Undefined by default
    padding = 'SAME'
    weight = None
    bias = None
    naming = None
    elu = False
    hole = None
    max_pooling_kernel_size = None  # Or integer
    max_pooling_strides = None      # Or integer
    res_type = RESIDUAL_NONE        # No residual
    gradb = False                   # bias with gradients
    initialized_as_zero = False     # Zero initial weights rather than randomized values
    batch_normalization = None      # input of tf.layers.batch_normalization(training=)
    cat = 'cnn'

    def __init__(self, ch_out, strides=[1, 2, 2, 1], kernel_size=[3,3], padding = 'SAME'):
        self.ch_out = ch_out
        self.strides = strides
        self.kernel_size = kernel_size
        self.padding = padding

    def create_kernel(self, prev_ch_out):
        '''
        Returns w,b
            w: a TENSOR of kernel
            b: a TENSOR of bias

            Note: with gradb, the returned b is a TENSOR rather than VARIABLE
        '''
        if self.weight is not None:
            return self.weight, self.bias
        ch_in = self.ch_in if self.ch_in > 0 else int(prev_ch_out)
        ch_out = self.ch_out
        [w, h] = self.kernel_size[0:2]
        d = 1.0 / np.sqrt(ch_in * w * h)
        weight_shape = [1, w, h, ch_in, ch_out]
        if not self.gradb:
            bias_shape = [self.ch_out]
        else:
            bias_shape = [self.ch_out, 3]
        # print('weight_shape {}'.format(weight_shape))
        # print('d {}'.format(d))
        if self.naming is None:
            if self.initialized_as_zero:
                weight = tf.Variable(tf.zeros(weight_shape))
                bias   = tf.Variable(tf.zeros(bias_shape))
            else:
                weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
                bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        else:
            if self.initialized_as_zero:
                initializer = tf.zeros_initializer()
            else:
                initializer = tf.random_uniform_initializer(-d, d)
            with tf.variable_scope(self.naming):
                weight = tf.get_variable("weight", weight_shape, tf.float32,
                        initializer)
                bias_name = "bias_elu" if self.elu else "bias"
                if self.gradb:
                    bias_name += "_gradb"
                bias = tf.get_variable(bias_name, bias_shape, tf.float32,
                        initializer)
        self.weight = weight
        self.bias = bias
        if self.res_type != RESIDUAL_NONE and (ch_in != ch_out or self.strides[1:3] != [1,1]):
            with tf.variable_scope(self.naming):
                d = 1.0 / np.sqrt(ch_in) # w = h = 1
                if self.initialized_as_zero:
                    initializer = tf.zeros_initializer()
                else:
                    initializer = tf.random_uniform_initializer(-d, d)
                ''' 1x1 conv '''
                res_proj = tf.get_variable("res_proj", [1,1,1] + [ch_in, ch_out],
                        tf.float32,
                        initializer)
            self.res_proj = res_proj
        else:
            self.res_proj = None
        return weight, bias

    def calc_gradb_tensor(self, bias, this_w, this_h):
        if not self.gradb:
            return None
        ch_out = self.ch_out
        ls_w = tf.linspace(0.0, this_w - 1.0, this_w)
        ls_h = tf.linspace(0.0, this_h - 1.0, this_h)
        ls_c = tf.linspace(1.0, 1.0, 1)
        X,Y,Z = tf.meshgrid(ls_w, ls_h, ls_c, indexing='ij')
        base, dx, dy = tf.split(bias, 3, axis=1)
        base_1 = tf.reshape(base, shape=[1,1,ch_out])
        dx_1 = tf.reshape(dx, shape=[1,1,ch_out])
        dy_1 = tf.reshape(dy, shape=[1,1,ch_out])
        bias = base_1 * Z + dx_1 * X + dy_1 * Y
        print('Grad bias shape {}'.format(bias.shape))
        return bias

    def apply_layer(self, prev_layer, residual):
        print('prev {}'.format(prev_layer.shape))
        prev_ch_out = int(prev_layer.shape[-1])
        prev_w, prev_h = int(prev_layer.shape[-3]),int(prev_layer.shape[-2])
        w,b = self.create_kernel(prev_ch_out)
        if self.hole is not None:
            strides = [1, self.strides[1], self.strides[2]]
            conv = tf.nn.convolution(prev_layer,
                    w,
                    padding=self.padding,
                    strides=strides,
                    dilation_rate=self.hole,
                    data_format='NDHWC')
        else:
            strides = [1] + self.strides
            conv = tf.nn.conv3d(prev_layer, w, strides, self.padding)
        # out = tf.nn.relu(conv + b)
        residual_out = None
        if self.res_type == RESIDUAL_JOIN:
            conv = conv + residual
            residual_out = None
        elif self.res_type == RESIDUAL_FORK:
            residual_out = prev_layer
        elif self.res_type == RESIDUAL_PASS:
            residual_out = residual
        if self.gradb:
            b = self.calc_gradb_tensor(b, int(conv.shape[-3]), int(conv.shape[-2]))
            print('CNN out shape {}'.format(conv.shape))
        luinput = conv + b
        if self.batch_normalization is not None:
            luinput = tf.layers.batch_normalization(luinput,
                    training=self.batch_normalization)
        out = tf.nn.elu(luinput) if self.elu else tf.nn.relu(luinput)
        if self.max_pooling_kernel_size is not None:
            ksize = [1, 1] + self.max_pooling_kernel_size + [1]
            strides = [1, 1] + self.max_pooling_strides + [1]
            out = tf.nn.max_pool3d(input=out,
                    ksize=ksize,
                    strides=strides,
                    padding=self.padding)
        print('after CNN {} res_type: {}'.format(out.shape, self.res_type))
        if self.res_proj is not None:
            residual_out = tf.nn.conv3d(residual_out,
                    self.res_proj,
                    [1,1]+self.strides[1:3]+[1], # Same strides as the conv
                    self.padding)
        if residual_out is not None:
            print('after CNN residual {}'.format(residual_out.shape))
        '''
        NOTE: Must return self.bias rather than b. b might be a tensor while self.bias store the variable.
        '''
        return w,self.bias,out,residual_out,self.res_proj

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
            if 'hole' in visdesc:
                # FIXME: More general
                rate = visdesc['hole']+1
                # Note: no hole in view dimension
                viscfg.hole = [1, rate, rate]
            if 'res' in visdesc:
                res_type = visdesc['res']
                if res_type == 'join':
                    viscfg.res_type = RESIDUAL_JOIN
                elif res_type == 'fork':
                    viscfg.res_type = RESIDUAL_FORK
                elif res_type == 'pass':
                    viscfg.res_type = RESIDUAL_PASS
                else:
                    raise NameError('Unknow residual string {}'.format(res_type))
            if 'max_pool' in visdesc:
                viscfg.max_pooling_kernel_size = visdesc['max_pool']['kernel_size']
                viscfg.max_pooling_strides = visdesc['max_pool']['strides']
            viscfg_array.append(viscfg)
        return viscfg_array

class FCLayerConfig:
    ch_in = -1
    ch_out = None
    weight = None
    bias = None
    naming = None
    elu = False
    gradb = False
    initialized_as_zero = False
    batch_normalization = None
    cat = 'fc'

    def __init__(self, ch_out):
        self.ch_out = ch_out
        self.nolu = False

    def create_kernel(self, prev_shape):
        if self.weight is not None:
            return self.weight, self.bias
        ch_in = self.ch_in if self.ch_in > 0 else int(prev_shape)
        ch_out = self.ch_out
        weight_shape = [ch_in, ch_out]
        bias_shape = [ch_out]
        d = 1.0 / np.sqrt(ch_in)
        if self.naming is None:
            if self.initialized_as_zero:
                weight = tf.Variable(tf.zeros(weight_shape))
                bias   = tf.Variable(tf.zeros(bias_shape))
            else:
                weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
                bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        else:
            if self.initialized_as_zero:
                initializer = tf.zeros_initializer()
            else:
                initializer = tf.random_uniform_initializer(-d, d)
            with tf.variable_scope(self.naming):
                weight = tf.get_variable("weight", weight_shape, tf.float32,
                        initializer)
                bias_name = "bias_elu" if self.elu else "bias"
                bias   = tf.get_variable(bias_name, bias_shape, tf.float32,
                        initializer)
        self.weight, self.bias = weight, bias
        return weight, bias

    def apply_layer(self, prev, _):
        # Dim 0 is BATCH
        print('prev {}'.format(prev.shape))
        if len(prev.shape.as_list()) > 2:
            flatten = tf.reshape(prev, [-1, prev.shape[1].value, reduce(mul, prev.shape[2:].as_list(), 1)])
            post_reshape = True
        else:
            flatten = prev
            post_reshape = False
        print('flatten {}'.format(flatten.shape))
        lastdim = len(flatten.shape.as_list()) - 1
        w,b = self.create_kernel(flatten.shape[-1])
        print('w,b {} {}'.format(w.get_shape(), b.get_shape()))
        td = tf.tensordot(flatten, w, [[lastdim], [0]])
        print('tdshape {}'.format(td.get_shape()))
        if post_reshape:
            td.set_shape([None, prev.shape[1].value, self.ch_out])
            print('tdshape set to {}'.format(td.get_shape()))
        if self.nolu:
            out = td + b
        else:
            luinput = td + b
            if self.batch_normalization is not None:
                luinput = tf.layers.batch_normalization(luinput,
                    training=self.batch_normalization)
            out = tf.nn.elu(luinput) if self.elu else tf.nn.relu(luinput)
        return w,b,out,None,None

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
    for tup in view_config:
        angle,ncam = tup[0:2]
        if len(tup) >= 3:
            base = tup[2]
        else:
            base = 0
        view_array += [ [angle,float(i)+base] for i in np.arange(0.0, 360.0, 360.0/float(ncam)) ]
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
    gradb = False

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
            elu=False,
            gradb=False,
            initialized_as_zero=False,
            nolu_at_final=False,
            batch_normalization=None):
        '''
        batch_normalization should be
         - None, disable batch_normalization
         - Boolean Scalar Tensor, enable batch_normalization, and feed this
           tensor to training= of tf.layers.batch_normalization
        '''
        self.layer_configs = []
        self.elu = elu
        self.gradb = gradb
        self.initialized_as_zero = initialized_as_zero
        if confdict is not None:
            self.layer_configs = VisionLayerConfig.createFromDict(confdict)
        if featnums is not None:
            for featnum in featnums:
                self.layer_configs.append(FCLayerConfig(featnum))
            if nolu_at_final:
                self.layer_configs[-1].nolu = True
        self.naming = naming
        if self.naming is not None:
            index = 0
            for conf in self.layer_configs:
                conf.naming = "Layer_{}".format(index)
                index += 1
        self.batch_normalization = batch_normalization
        self.cat_nn_vars = None
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
            layer.gradb = self.gradb
            layer.initialized_as_zero = self.initialized_as_zero
            layer.batch_normalization = self.batch_normalization
        nn_layer_tensor = [input_tensor]
        nn_args = []
        if self.cat_nn_vars is None:
            # Note: do NOT initialize self.cat_nn_vars right now
            #       We need self.cat_nn_vars is None to indicate this is the
            #       first time to run infer_impl
            cat_nn_vars = dict()
        residual = None
        # print("== Len of layer_configs {}".format(len(self.layer_configs)))
        for conf in self.layer_configs:
            prev_layer_tensor = nn_layer_tensor[-1]
            w,b,out,residual,res_proj = conf.apply_layer(prev_layer_tensor, residual)
            nn_layer_tensor.append(out)
            if res_proj is None:
                current_layer_args = [w,b]
            else:
                current_layer_args = [w,b,res_proj]
            nn_args += current_layer_args
            if self.cat_nn_vars is None:
                if conf.cat not in cat_nn_vars:
                    cat_nn_vars[conf.cat] = []
                cat_nn_vars[conf.cat] += current_layer_args
        if self.cat_nn_vars is None:
            self.cat_nn_vars = cat_nn_vars
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


def MergeSVFeatVec(tensor):
    '''
    Input: [B, V, F]
    Output: [B, 1, V*F]

    B may be -1
    '''
    V,F = int(tensor.shape[1]), int(tensor.shape[2])
    # print("V,F {} {}".format(V, F))
    return tf.reshape(tensor, shape=[-1, 1, V*F])

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
            combine_in = MergeSVFeatVec(tf.concat([rgb_nn_featvec, depth_nn_featvec], axis=-1))
            print("combine_in shape {}".format(combine_in.shape))
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
            combine_params, combine_out = self.mv_fc_applier.infer(MergeSVFeatVec(rgbd_nn_featvec))
        self.reuse = True
        return rgbd_nn_params + combine_params, combine_out

'''
Feature Extractor Rev. 4
Arch:
    SVCNN(rgb,dep) := (CNN^m) (cat(rgb,dep))
    FE := FC^N * stack * FC^K * SVFE
    output = FE(rgb, dep)
'''
class FeatureExtractorRev4:
    naming = None
    reuse = None
    sv_non_shared = None
    viewnum = 0

    def __init__(self,
            sv_shared_conf,
            sv_nonshared_conf,
            viewnum,
            sv_featnums,
            mv_featnums,
            naming,
            elu):
        self.sv_shared = ConvApplier(sv_shared_conf, None, "PerViewSharedRGBD", elu)
        self.sv_non_shared = []
        for i in range(viewnum):
            self.sv_non_shared.append(ConvApplier(sv_nonshared_conf,
                sv_featnums, "View{}_NonSharedRGBD".format(i), elu))
        self.mv_fc_applier = ConvApplier(None, mv_featnums, "RGBDCombineViewFC", elu)

        self.viewnum = viewnum
        self.naming = naming

    def infer(self, rgb_input, depth_input):
        with tf.variable_scope(self.naming, reuse=self.reuse) as scope:
            rgbd_input = tf.concat([rgb_input, depth_input], axis=-1)
            rgbd_shared_params, rgbd_shared_featvec = self.sv_shared.infer(rgbd_input)
            perview_intermediates = tf.split(rgbd_input, [1 for i in range(self.viewnum)], axis=1)
            perview_params = []
            perview_featvec = []
            for i in range(self.viewnum):
                params, featvec = self.sv_non_shared[i].infer(perview_intermediates[i])
                perview_params.append(params)
                perview_featvec.append(featvec)
            print("> perview featvec shape {}".format(perview_featvec[-1].shape))
            combine_in = tf.concat(perview_featvec, axis=2)
            combine_params, combine_out = self.mv_fc_applier.infer(combine_in)
        self.reuse = True
        return rgbd_shared_params + perview_params + combine_params, combine_out

'''
Feature Extractor Rev. 5
Arch:
    AlexNet
'''
class FeatureExtractorRev5:
    naming = None
    reuse = None

    def __init__(self,
            cnnconf,
            featnums,
            naming,
            elu):
        self.alexrgbd = ConvApplier(cnnconf, featnums, "AlexRGBD", elu)

        self.naming = naming

    def infer(self, rgb_input, depth_input):
        with tf.variable_scope(self.naming, reuse=self.reuse) as scope:
        #with tf.variable_scope(self.naming, reuse=tf.AUTO_REUSE) as scope:
            rgbd_input = tf.concat([rgb_input, depth_input], axis=-1)
            params, featvec = self.alexrgbd.infer(rgbd_input)
            print("> Rev5 {}".format(featvec.shape))
        self.reuse = True
        return params, featvec

'''
Feature Extractor Rev. 6
Arch:
    Dilated convolution
'''
class FeatureExtractorRev6:
    naming = None
    reuse = None

    def __init__(self,
            cnnconf,
            featnums,
            naming,
            elu):
        self.alexrgbd = ConvApplier(cnnconf, featnums, "HoleRGBD", elu)

        self.naming = naming

    def infer(self, rgb_input, depth_input):
        with tf.variable_scope(self.naming, reuse=self.reuse) as scope:
        #with tf.variable_scope(self.naming, reuse=tf.AUTO_REUSE) as scope:
            rgbd_input = tf.concat([rgb_input, depth_input], axis=-1)
            params, featvec = self.alexrgbd.infer(rgbd_input)
            print("> Rev6 {}".format(featvec.shape))
        self.reuse = True
        return params, featvec


'''
Feature Extractor ResNet
Arch:
    ResNet18
'''
class FeatureExtractorResNet:
    naming = None
    reuse = None

    def __init__(self,
            cnnconf,
            featnums,
            naming,
            elu,
            gradb=False,
            batch_normalization=None):
        self.resnetrgbd = ConvApplier(cnnconf, featnums, "ResNet18RGBD", elu, gradb,
                batch_normalization=batch_normalization)

        self.naming = naming

    def infer(self, rgb_input, depth_input):
        with tf.variable_scope(self.naming, reuse=self.reuse) as scope:
        #with tf.variable_scope(self.naming, reuse=tf.AUTO_REUSE) as scope:
            rgbd_input = tf.concat([rgb_input, depth_input], axis=-1)
            params, featvec = self.resnetrgbd.infer(rgbd_input)
            self.cat_nn_vars = self.resnetrgbd.cat_nn_vars.copy()
            print("> Rev11 {}".format(featvec.shape))
        self.reuse = True
        return params, featvec

