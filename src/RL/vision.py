import tensorflow as tf
import numpy as np
from operator import mul

class VisionLayerConfig:
    strides = [1, 2, 2, 1]
    kernel_size = [3, 3]
    ch_in = -1                      # -1 means all channels from previous layer
    ch_out = None                   # Undefined by default
    padding = 'SAME'

    def __init__(self, ch_out, strides=[1, 2, 2, 1], kernel_size=[3,3], padding = 'SAME'):
        self.ch_out = ch_out
        self.strides = strides
        self.kernel_size = kernel_size
        self.padding = padding

    def create_kernel(self, prev_ch_out):
        ch_in = self.ch_in if self.ch_in > 0 else int(prev_ch_out)
        ch_out = self.ch_out
        [w, h] = self.kernel_size[0:2]
        d = 1.0 / np.sqrt(ch_in * w * h)
        weight_shape = [1, w, h, ch_in, ch_out]
        bias_shape = [self.ch_out]
        # print('weight_shape {}'.format(weight_shape))
        # print('d {}'.format(d))
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

    def apply_layer(self, prev_layer):
        prev_ch_out = int(prev_layer.shape[-1])
        w,b = self.create_kernel(prev_ch_out)
        strides = [1] + self.strides
        conv = tf.nn.conv3d(prev_layer, w, strides, self.padding)
        out = tf.nn.relu(conv + b)
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

    def __init__(self, ch_out):
        self.ch_out = ch_out

    def create_kernel(self, prev_shape):
        ch_in = self.ch_in if self.ch_in > 0 else int(prev_shape)
        ch_out = self.ch_out
        weight_shape = [ch_in, ch_out]
        bias_shape = [ch_out]
        d = 1.0 / np.sqrt(ch_in)
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

    def apply_layer(self, prev):
        # Dim 0 is BATCH
        print('prev {}'.format(prev.shape))
        flatten = tf.reshape(prev, [-1, prev.shape[1].value, reduce(mul, prev.shape[2:].as_list(), 1)])
        print('flatten {}'.format(flatten.shape))
        w,b = self.create_kernel(flatten.shape[-1])
        print('w,b {} {}'.format(w.get_shape(), b.get_shape()))
        td = tf.tensordot(flatten, w, [[2], [0]])
        print('tdshape {}'.format(td.get_shape()))
        td.set_shape([None, self.ch_out])
        print('tdshape set to {}'.format(td.get_shape()))
        return w,b,tf.nn.relu(td + b)

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
    nn_layers = [] # 0 is input, -1 is output
    nn_args = []
    nn_filters = []
    nn_featvec = None

    def __init__(self,
            input_shape,
            layer_configs,
            thread_index,
            feature_number = -1,
            input_tensor = None):
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

    def get_output_tensors(self):
        if not self.nn_layers:
            self.infer()
        return self.nn_featvec

    features = property(get_output_tensors)

    def infer(self):
        self.nn_layers = [self.input_tensor]
        self.nn_args = []
        self.nn_filters = []
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

