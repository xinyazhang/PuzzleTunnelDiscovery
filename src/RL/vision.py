import tensorflow as tf
import numpy as np

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
        weight_shape = [w, h, ch_in, ch_out]
        bias_shape = [self.ch_out]
        # print('weight_shape {}'.format(weight_shape))
        # print('d {}'.format(d))
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

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
        flatten = tf.reshape(prev, [1, -1])
        w,b = self.create_kernel(flatten.shape[-1])
        return w,b,tf.nn.relu(tf.matmul(flatten, w) + b)

class VisionNetwork:
    thread_index = 0
    layer_number = 0
    input_placeholders = None
    layer_configs = None
    view_number = 0
    feature_number = -1
    nn_layers = [] # 0 is input, -1 is output
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
        self.view_number = int(input_shape[0])

    def get_output_tensors(self):
        if not self.nn_layers:
            self.infer()
        return self.nn_featvec

    features = property(get_output_tensors)

    def infer(self):
        nviews = int(self.input_tensor.shape[0])
        self.nn_layers = [self.input_tensor]
        self.nn_filters = []
        for conf in self.layer_configs:
            prev_layer = self.nn_layers[-1]
            print("prev shape {}".format(prev_layer.shape))
            prev_ch = prev_layer.shape[-1]
            w,b = conf.create_kernel(prev_ch)
            conv = tf.nn.conv2d(prev_layer, w, conf.strides, conf.padding)
            out = tf.nn.relu(conv + b)
            self.nn_layers.append(out)
        if self.feature_number < 0:
            self.nn_featvec = tf.reshape(self.nn_layers[-1], [-1])
            return
        fclc = FCLayerConfig(self.feature_number)
        self.mv_cnnflatten = tf.reshape(self.nn_layers[-1], [self.view_number, -1])
        print('mv_cnn flatten: {}'.format(self.mv_cnnflatten.shape))
        #w,b = fclc.create_kernel(self.mv_cnnflatten.shape[-1])
        #self.mv_featvec = tf.nn.relu(tf.matmul(self.mv_cnnflatten, w) + b)
        w,b,self.mv_featvec = fclc.apply_layer(self.mv_cnnflatten)
        self.nn_featvec = self.mv_featvec

