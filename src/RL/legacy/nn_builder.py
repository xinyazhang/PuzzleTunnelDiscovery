import tensorflow as tf
import numpy as np
from operator import mul

RESIDUAL_NONE = 0
RESIDUAL_FORK = 1
RESIDUAL_PASS = 2
RESIDUAL_JOIN = 3

GRADBIAS_ELEMENTS_PER_CHANNEL = 3

'''
Should be CNNLayerConfig
'''
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
    avg_pooling_kernel_size = None  #
    avg_pooling_strides = None      #
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
        if self.avg_pooling_kernel_size is not None:
            ksize = [1, 1] + self.avg_pooling_kernel_size + [1]
            strides = [1, 1] + self.avg_pooling_strides + [1]
            out = tf.nn.avg_pool3d(input=out,
                    ksize=ksize,
                    strides=strides,
                    padding='VALID')
            print('after avg_pool3d shape {}'.format(out.shape))
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

    '''
    FIXME: Deprecated, use CreateVisionLayersFromDict instead
    @staticmethod
    def createFromDict(visdict):
        return CreateVisionLayersFromDict(visdict)
    '''



def CreateVisionLayersFromDict(visdict):
    viscfg_array = []
    CONSTRUCTOR = VisionLayerConfig
    for visdesc in visdict:
        print(visdesc)
        viscfg = CONSTRUCTOR(visdesc['ch_out'])
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
        if 'avg_pool' in visdesc:
            viscfg.avg_pooling_kernel_size = visdesc['avg_pool']['kernel_size']
            viscfg.avg_pooling_strides = visdesc['avg_pool']['strides']
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
            self.layer_configs = CreateVisionLayersFromDict(confdict)
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

    def infer(self, input_tensor, return_per_layer_tensor=False):
        if self.naming is None:
            return self.infer_impl(input_tensor)
        else:
            with tf.variable_scope(self.naming):
                return self.infer_impl(input_tensor, return_per_layer_tensor)

    def infer_impl(self, input_tensor, return_per_layer_tensor):
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
        if return_per_layer_tensor:
            return nn_args, nn_layer_tensor[-1], nn_layer_tensor
        return nn_args, nn_layer_tensor[-1]


class TransposeCNNConfig(object):
    padding = 'SAME'
    weight = None
    bias = None
    naming = None
    elu = False
    max_pooling_kernel_size = None  # Or integer
    max_pooling_strides = None      # Or integer
    res_type = RESIDUAL_NONE        # No residual
    initialized_as_zero = False     # Zero initial weights rather than randomized values
    batch_normalization = None      # input of tf.layers.batch_normalization(training=)
    cat = 'cnn'

    def __init__(self, dic_entry, enc_input, enc_output):
        self.output_shape = enc_input.shape.as_list()
        self.input_shape = enc_output.shape.as_list()
        self.dic_entry = dic_entry
        self.strides = dic_entry['strides']
        if self.strides is None:
            self.strides = [1,2,2,1]
        self.kernel_size = dic_entry['kernel_size']
        if self.kernel_size is None:
            self.kernel_size = [3,3]
        if 'max_pool' in dic_entry:
            ch_in = self.input_shape[-1]
            self.max_pooling_kernel_size = [1] + dic_entry['max_pool']['kernel_size'] + [ch_in, ch_in]
            self.max_pooling_strides = [1,1] + dic_entry['max_pool']['strides'] + [1]
        if 'res' in dic_entry:
            if 'join' == dic_entry['res']:
                self.res_type = RESIDUAL_JOIN
            elif 'fork' == dic_entry['res']:
                self.res_type = RESIDUAL_FORK
        else:
            self.res_type = RESIDUAL_PASS

    def create_kernel(self, prev_ch_out):
        assert self.naming, 'naming is mandantory for TransposeCNNConfig'
        assert self.initialized_as_zero is False, 'TransposeCNNConfig prohibits zero initialization'

        if self.weight is not None:
            return self.weight, self.bias
        ch_in = self.input_shape[-1]
        ch_out = self.output_shape[-1]
        w,h = self.kernel_size
        weight_shape = [1, w, h, ch_out, ch_in] # IMPORTANT, out before in

        initializer = tf.random_uniform_initializer(-d, d)
        with tf.variable_scope(self.naming):
            weight = tf.get_variable("weight", weight_shape, tf.float32,
                    initializer)
            bias_name = "bias_elu" if self.elu else "bias"
            bias = tf.get_variable(bias_name, bias_shape, tf.float32,
                    initializer)
            if self.max_pooling_kernel_size is not None:
                self.upsampling_deconv = tf.get_variable('unpooling',
                        self.max_pooling_kernel_size, tf.float32,
                        initializer)
            else:
                self.upsampling_deconv = None

            # Projection
            if self.res_type != RESIDUAL_NONE and (ch_in != ch_out or self.strides[1:3] != [1,1]):
                with tf.variable_scope(self.naming):
                    d = 1.0 / np.sqrt(ch_in) # w = h = 1
                    initializer = tf.random_uniform_initializer(-d, d)
                    ''' 1x1 conv '''
                    res_unproj = tf.get_variable("res_proj_transpose", [1,1,1] + [ch_out, ch_in],
                            tf.float32,
                            initializer)
                self.res_unproj = res_unproj
            else:
                self.res_unproj = None
        self.weight = weight
        self.bias = bias

        return weight, bias

    def apply_layer(self, prev_layer, residual):
        w,b = self.create_kernel()
        assert self.hole is False, 'dilated convolution is not supported by TransposeCNNConfig'
        in_layer = prev_layer
        if self.upsampling_deconv is not None:
            upsampling_deconv_shape = list(self.input_shape)
            upsampling_deconv_shape[2] *= 2
            upsampling_deconv_shape[3] *= 2
            upsampled = tf.nn.conv3d_transpose(prev_layer,
                    self.unpooling_deconv,
                    output_shape=upsampling_deconv_shape,
                    strides=self.max_pooling_strides,
                    padding='SAME')
            print('Upsampling through conv3d_transpose from {} to {}'.format(prev_layer.shape,
                upsampled.shape))
            in_layer = upsampled

        conv = tf.nn.conv3d_transpose(in_layer, w,
                output_shape=self.output_shape,
                strides=[1] + self.strides, padding=self.padding)
        residual_out = None
        if self.res_type == RESIDUAL_JOIN:
            residual_out = prev_layer # Inverse order, treat Join as Fork
        elif self.res_type == RESIDUAL_FORK:
            conv += residual # Inverse order, treat Fork as Join
            residual_out = None
        elif self.res_type == RESIDUAL_PASS:
            residual_out = residual # Unchanged
        luinput = conv + b
        if self.batch_normalization is not None:
            luinput = tf.layers.batch_normalization(luinput,
                    training=self.batch_normalization)
        out = tf.nn.elu(luinput) if self.elu else tf.nn.relu(luinput)
        assert self.avg_pooling_kernel_size is None, 'avg_pool is unsupported by TransposeCNNConfig'
        print('after CNN {} res_type: {}'.format(out.shape, self.res_type))
        if self.res_proj is not None:
            residual_out = tf.nn.conv3d_transpose(residual_out,
                    self.res_proj,
                    output_shape, # Should match the conventional deconv
                    strides=[1] + self.strides,
                    padding=self.padding)
        if residual_out is not None:
            print('after CNN residual {}'.format(residual_out.shape))
        return w,self.bias,out,residual_out,self.res_proj

#
# TODO: IMPLEMENT THIS
class DeFCLayerConfig(object):
    naming = None

    def __init__(self, enc_input, enc_output):
        self.output_shape = enc_input.shape.as_list()
        self.input_shape = enc_output.shape.as_list()
        assert self.input_shape[0:2] == [-1,1], "DeFCLayerConfig assumes feature vector merged from multiple views"
        self.featnum = self.input_shape[-1]
        self.mat_output_size = reduce(mul, self.output_shape[2:]) # size of per-view feature maps
        self.weight = None
        self.bias = None
        self.elu = True

    def create_kernel(self):
        if self.weight is not None:
            return self.weight, self.bias
        view = self.output_shape[1]
        ch_in = self.featnum
        ch_out = self.mat_output_size
        weight_shape = [view, ch_in, ch_out]
        bias_shape = [view, ch_out]
        assert self.naming is not None
        initializer = tf.random_uniform_initializer(-d, d)
        with tf.variable_scope(self.naming):
            weight = tf.get_variable("decoder_weight", weight_shape, tf.float32,
                    initializer)
            bias_name = "decoder_bias_elu" if self.elu else "decoder_bias"
            bias = tf.get_variable(bias_name, bias_shape, tf.float32,
                    initializer)
        self.weight, self.bias = weight, bias
        return weight, bias

    def apply_layer(self, prev, _):
        assert prev.shape.as_list() == self.input_shape
        prev_2d = tf.reshape(prev, [-1, self.featnum])
        w,b = self.create_kernel(flatten.shape[-1])
        td = tf.tensordot(prev_2d, w, [[1], [1]])
        V = self.output_shape[1]
        M = self.mat_output_size
        assert td.shape.as_list() = [-1, V, M], 'Fatal shape {}'.format(td.shape.as_list())
        out = tf.nn.elu(td + b)
        out = tf.reshape(out, self.output_shape)
        return w,b,out,None,None

class DeconvApplier:
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
            naming,
            encoder_layers,
            elu=False,
            nolu_at_final=False,
            batch_normalization=None):
        assert confdict is not None, 'confdict must not be None for DeconvApplier'
        '''
        batch_normalization should be
         - None, disable batch_normalization
         - Boolean Scalar Tensor, enable batch_normalization, and feed this
           tensor to training= of tf.layers.batch_normalization
        '''
        self.elu = elu
        self.encoder_layers = encoder_layers
        self.layer_configs = []
        for dic_entry, (enc_input, enc_output) in zip(confdict, zip(encoder_layers, encoder_layers[1:])):
            self.layer_configs.append(TransposeCNNConfig(dic_entry, enc_input, enc_output))
        defc = DeFCLayerConfig(encoder_layers[-2], encoder_layers[-1])
        self.layer_configs = [defc] + self.layer_configs[::-1]
        # Shift the ch_out
        # For deconvolution the channel output should be the
        self.naming = naming
        if self.naming is not None:
            index = 0
            for conf in self.layer_configs:
                conf.naming = "Transpose_Layer_{}".format(index)
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

