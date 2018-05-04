from abc import ABCMeta, abstractmethod, abstractproperty
import tensorflow as tf
from collections import deque
import itertools
import random

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

class IExperienceReplayEnvironment(IEnvironment):

    def __init__(self, tmax, erep_cap):
        super(IExperienceReplayEnvironment, self).__init__()
        self.tmax = tmax
        self.erep_sample_cap = erep_cap * tmax
        self.erep_actions = deque()
        self.erep_states = deque()
        self.erep_reward = deque()
        self.erep_term = deque()
        self.erep_all_deques = [self.erep_actions, self.erep_states,
                self.erep_reward, self.erep_term]
        '''
        Actionn State Reward Queues
        '''
        self.erep_asr_deques = [self.erep_actions, self.erep_states, self.erep_reward]

    '''
    Store Experience REPlay
    '''
    def store_erep(self, action, state, reward, reaching_terminal):
        if self.erep_sample_cap <= 0:
            return
        while len(self.erep_actions) >= self.erep_sample_cap:
            [q.popleft() for q in self.erep_all_deques]
        self.erep_actions.append(action)
        self.erep_states.append(state)
        self.erep_reward.append(reward)
        self.erep_term.append(reaching_terminal)

    def sample_in_erep(self, pprefix):
        # FIXME: Handle terminal in the cache
        cached_samples = len(self.erep_actions)
        size = min(cached_samples, self.tmax)
        start = random.randrange(-size + 1, cached_samples)
        pick_start = max(start, 0)
        pick_end = min(start + size, cached_samples)
        a,s,r = [list(itertools.islice(q, pick_start, pick_end)) for q in self.erep_asr_deques]
        '''
        Number of actions = Number of States - 1
        '''
        a=a[:-1]
        r=r[:-1]
        return a,s,r,self.erep_term[pick_end-1]

    def reset(self):
        [q.clear() for q in self.erep_all_deques]

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
    vstates: List of vstate from IEnvironment
    '''
    @abstractmethod
    def evaluate(self, vstates, sess, tensors, additional_dict=None):
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
    Return the artificial reward
    '''
    @abstractmethod
    def get_artificial_from_experience(self, sess, vstates, action_performed):
        pass

    '''
    Train/Refine any embedded model

    Note: cannot be merged to a2c since ICM has different inputs
    Note2: actions should be a distribution

    CAVEAT: deprecated, use build_loss instead
    '''
    def train(self, sess, rgb, dep, actions):
        raise "Deprecated buggy solution"

    @abstractmethod
    def build_loss(self):
        '''
        Return the interal loss function from a segment
        '''
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
