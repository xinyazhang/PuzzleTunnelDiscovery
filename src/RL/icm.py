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
            elu,
            ferev=1):
        self.action_tensor = action_tensor
        self.rgb_tensor = rgb_tensor
        self.depth_tensor = depth_tensor
        self.next_rgb_tensor = next_rgb_tensor
        self.next_depth_tensor = next_depth_tensor
        if ferev == 1:
            self.feature_extractor = vision.FeatureExtractor(svconfdict, mvconfdict, featnum, featnum, 'VisionNet', elu)
        elif ferev == 2:
            self.feature_extractor = vision.FeatureExtractorRev2(svconfdict,
                    128, [featnum * 2, featnum], 'VisionNetRev2', elu)
        elif ferev == 3:
            self.feature_extractor = vision.FeatureExtractorRev3(svconfdict,
                    128, [featnum * 2, featnum], 'VisionNetRev3', elu)
        elif ferev == 4:
            self.feature_extractor = vision.FeatureExtractorRev4(
                    config.SV_SHARED,
                    config.SV_NON_SHARED,
                    int(rgb_tensor.shape[1]),
                    [128, 128],
                    [featnum * 2, featnum], 'VisionNetRev4', elu)
        elif ferev == 5:
            self.feature_extractor = vision.FeatureExtractorRev5(
                    config.SV_NAIVE,
                    [1024, 1024, featnum], 'VisionNetRev5', elu)
        elif ferev == 6:
            self.feature_extractor = vision.FeatureExtractorRev6(
                    config.SV_HOLE_LOWRES,
                    [1024, 1024, featnum], 'VisionNetRev6', elu)
        elif ferev == 7:
            self.feature_extractor = vision.FeatureExtractorRev6(
                    config.SV_HOLE_MIDRES,
                    [1024, 1024, featnum], 'VisionNetRev7', elu)
        elif ferev == 8:
            self.feature_extractor = vision.FeatureExtractorRev6(
                    config.SV_HOLE_HIGHRES,
                    [1024, 1024, featnum], 'VisionNetRev8', elu)
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
        # TODO: with tf.squeeze ?
        '''
        ret = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.action_tensor,
            logits=out)
        '''
        labels = tf.reshape(self.action_tensor, [-1, 12])
        logits = tf.reshape(out, [-1, 12])
        ret = tf.losses.softmax_cross_entropy(
            onehot_labels=labels,
            logits=logits)
        '''
        ret = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=labels,
            logits=logits)
        '''
        print('inv loss ret shape {}'.format(ret.shape))
        return ret

'''
IntrinsicCuriosityModuleCommittee:
    ICM Committe, each NN for one view
    Inverse model: accumulating prediction from views as multi-view prediction.
    Forward model: TODO
'''
class IntrinsicCuriosityModuleCommittee:
    icms = None
    view_num = 0
    inverse_output_tensor = None
    forward_output_tensor = None

    @staticmethod
    def scope_name(i):
        return 'ICM_View{}'.format(i)

    def __init__(self,
            action_tensor,
            rgb_tensor,
            depth_tensor,
            next_rgb_tensor,
            next_depth_tensor,
            svconfdict,
            mvconfdict,
            featnum,
            elu,
            ferev=1):
        self.icms = []
        self.view_num = int(rgb_tensor.shape[1])
        self.perview_rgbs_1 = tf.split(rgb_tensor, self.view_num, axis=1)
        self.perview_deps_1 = tf.split(depth_tensor, self.view_num, axis=1)
        self.perview_rgbs_2 = tf.split(next_rgb_tensor, self.view_num, axis=1)
        self.perview_deps_2 = tf.split(next_depth_tensor, self.view_num, axis=1)
        self.action_tensor = action_tensor
        for i in range(self.view_num):
            with tf.variable_scope(scope_name(i)):
                self.icms.append(IntrinsicCuriosityModule(
                    action_tensor,
                    self.perview_rgbs_1[i],
                    self.perview_deps_1[i],
                    self.perview_rgbs_2[i],
                    self.perview_deps_2[i],
                    svconfdict,
                    mvconfdict,
                    elu,
                    ferev))

    def get_inverse_model(self):
        if self.inverse_output_tensor is not None:
            return self.inverse_model_params, self.inverse_output_tensor
        paramss = []
        outs = []
        for i in range(self.view_num):
            icm = self.icms[i]
            with tf.variable_scope(scope_name(i)):
                params, out = icm.get_inverse_model()
                paramss.append(params)
                outs.append(out)
        self.inverse_model_params = sum(paramss, [])
        self.inverse_output_tensor = tf.accumulate_n(outs)
        return self.inverse_model_params, self.inverse_output_tensor

    def get_forward_model(self):
        if self.forward_output_tensor is not None:
            return self.forward_model_params, self.forward_output_tensor
        pass

    def get_inverse_loss(self):
        _, out = self.get_inverse_model()
        print('> inv loss out.shape {}'.format(out.shape))
        print('> inv loss action.shape {}'.format(out.shape))
        labels = tf.reshape(self.action_tensor, [-1, 12])
        logits = tf.reshape(out, [-1, 12])
        ret = tf.losses.softmax_cross_entropy(
            onehot_labels=labels,
            logits=logits)
        print('inv loss ret shape {}'.format(ret.shape))
        return ret
