import tensorflow as tf
import numpy as np
import vision
import config

class IntrinsicCuriosityModule:
    action_tensor = None
    rgb_tensor = None
    depth_tensor = None
    next_rgb_tensor = None
    next_depth_tensor = None
    feature_extractor = None
    inverse_fc_applier = None
    inverse_model_params = None
    inverse_output_tensor = None
    forward_model_params = None
    forward_output_tensor = None

    '''
    Persumably these tensors shall be placeholders
    '''
    def __init__(self,
            action_tensor,
            rgb_tensor,
            depth_tensor,
            next_rgb_tensor,
            next_depth_tensor,
            svconfdict,
            mvconfdict,
            featnum,
            elu):
        self.action_tensor = action_tensor
        self.rgb_tensor = rgb_tensor
        self.depth_tensor = depth_tensor
        self.next_rgb_tensor = next_rgb_tensor
        self.next_depth_tensor = next_depth_tensor
        self.feature_extractor = vision.FeatureExtractor(svconfdict, mvconfdict, featnum, featnum, 'VisionNet', elu)
        self.cur_nn_params, self.cur_featvec = self.feature_extractor.infer(rgb_tensor, depth_tensor)
        self.next_nn_params, self.next_featvec = self.feature_extractor.infer(next_rgb_tensor, next_depth_tensor)
        self.elu = elu

    def get_inverse_model(self):
        if self.inverse_output_tensor is not None:
            return self.inverse_model_params, self.inverse_output_tensor
        input_featvec = tf.concat([self.cur_featvec, self.next_featvec], 2)
        print('inverse_model input {}'.format(input_featvec))
        featnums = [config.INVERSE_MODEL_HIDDEN_LAYER, int(self.action_tensor.shape[-1])]
        print('inverse_model featnums {}'.format(featnums))
        self.inverse_fc_applier = vision.ConvApplier(None, featnums, 'InverseModelNet', self.elu)
        params, out = self.inverse_fc_applier.infer(input_featvec)
        self.inverse_model_params = params
        self.inverse_output_tensor = out
        return params, out

    def get_forward_model(self):
        if self.forward_output_tensor is not None:
            return self.forward_model_params, self.forward_output_tensor
        '''
        action_tensor has shape [None, N], but our pipeline usually use [None, 1, N]

        Note: 3D tensor unifies per-view variables and combined-view variables.
        '''
        input_featvec = tf.concat([self.action_tensor, self.cur_featvec], 2)
        featnums = list(config.FORWARD_MODEL_HIDDEN_LAYERS) + [int(self.action_tensor.shape[-1])]
        self.forward_fc_applier = vision.ConvApplier(None, featnums, 'ForwardModelNet', self.elu)
        params, out = self.forward_fc_applier.infer(input_featvec)
        self.forward_model_params = params
        self.forward_output_tensor = out
        return params, out

    def get_nn_params(self):
        ret = [self.cur_nn_params]
        params, _ = self.get_inverse_model()
        ret.append(params)
        params, _ = self.get_forward_model()
        ret.append(params)
        return sum(ret, [])

    def get_inverse_loss(self, discrete=False):
        _, out = self.get_inverse_model()
        print('inv loss out.shape {}'.format(out.shape))
        print('inv loss action.shape {}'.format(out.shape))
        if not discrete:
            return tf.norm(out - self.action_tensor)
        ret = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.action_tensor,
            logits=out)
        print('inv loss ret shape {}'.format(ret.shape))
        return tf.reduce_mean(ret)