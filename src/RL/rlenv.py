from abc import ABCMeta, abstractmethod, abstractproperty
import tensorflow as tf

'''
Note: Python2 code?
'''

class IEnvironment(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(IEnvironment, self).__init__()

    '''
    Property for Configuration(Q) state
    RW
    '''
    @abstractmethod
    def qstate_setter(self, state):
        pass

    @abstractmethod
    def qstate_getter(self):
        pass

    qstate = abstractproperty(qstate_getter, qstate_setter)

    '''
    Property for Viewable state (i.e. RGB-D images)
    RO, should return [rgb,dep]
    '''
    @abstractproperty
    def vstate(self):
        pass

    '''
    Property for Viewable state (i.e. RGB-D images) dimension
    RO, should return [view,w,h]
    '''
    @abstractproperty
    def vstatedim(self):
        pass

    '''
    Return [new_state, reward, reaching_terminal]
    Note:
        1. do not actually perform the action -- use qstate setter to do this
        2. artificial reward is in IAdvantageCore
    '''
    @abstractmethod
    def peek_act(self, action):
        pass

    '''
    Reset to initial state
    '''
    @abstractmethod
    def reset(self):
        pass

'''
Overall design idea:
    separate the TF inferencing network, training network, rendering, and TF session
    - Inferencing network is managed by itself
    - training network is managed by trainer classes (e.g. A2CTrainer)
    - rendering is moved to IEnvironment
    - TF session is provided by trainer class
'''
class IAdvantageCore(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(IAdvantageCore, self).__init__()
        self._softmax_policy_tensor = None

    '''
    Input RGB for Current Frame
    RO, placeholder
    '''
    @abstractproperty
    def rgb_1(self):
        pass

    '''
    Input RGB for Next Frame
    RO, placeholder
    '''
    @abstractproperty
    def rgb_2(self):
        pass

    '''
    Input Depth for Current Frame
    RO, placeholder
    '''
    @abstractproperty
    def dep_1(self):
        pass

    '''
    Input Depth for Next Frame
    RO, placeholder
    '''
    @abstractproperty
    def dep_2(self):
        pass

    '''
    Policy tensor (action distribution) before softmax
    Input placeholders are rgb_1 and dep_1

    RO, tensor
    '''
    @abstractproperty
    def policy(self):
        pass

    @property
    def softmax_policy(self):
        if self._softmax_policy_tensor is None:
            self._softmax_policy_tensor = tf.nn.softmax(logits=self.policy)
        return self._softmax_policy_tensor

    '''
    Value tensor (Q function value)
    Input placeholders are rgb_1 and dep_1

    RO, tensor
    '''
    @abstractproperty
    def value(self):
        pass

    '''
    NN Params of Policy Net
    RO, list of tf.Variable
    '''
    @abstractproperty
    def policy_params(self):
        pass

    '''
    NN Params of Value Net
    RO, list of tf.Variable
    '''
    @abstractproperty
    def value_params(self):
        pass

    '''
    Evaluate the environment
    '''
    @abstractmethod
    def evaluate_current(self, envir, sess, tensors, additional_dict=None):
        pass

    '''
    Return the index of decided action from a policy distribution
    Note: due to the interface of cross entropy loss,
          the input was not filtered by softmax
    Note2: we need envir because we want to store agent specific data to envir
    '''
    @abstractmethod
    def make_decision(self, envir, policy_dist):
        pass

    '''
    Return the artificial reward
    '''
    @abstractmethod
    def get_artificial_reward(self, envir, sess, state_1, adist, state_2):
        pass

    '''
    Train/Refine any embedded model

    Note: cannot be merged to a2c since ICM has different inputs
    Note2: actions should be a distribution
    '''
    @abstractmethod
    def train(self, sess, rgb, dep, actions):
        pass

    '''
    Return the cached lstm state for the next state.
    '''
    @abstractproperty
    def lstm_next(self):
        pass

    @abstractmethod
    def set_lstm(self, lstm):
        pass

    @abstractmethod
    def get_lstm(self):
        pass
