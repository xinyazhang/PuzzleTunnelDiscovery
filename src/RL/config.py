# -*- coding: utf-8 -*-
import numpy as np

class Config:
    LOCAL_T_MAX = 20            # repeat step size
    RMSP_ALPHA = 0.99           # decay parameter for RMSProp
    RMSP_EPSILON = 0.1          # epsilon parameter for RMSProp
    CHECKPOINT_DIR = 'puzzle_rl_checkpoints'
    LOG_FILE = 'tmp/puzzle_a3c_log'
    INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
    INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate
    INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
    GAMMA = 0.99                # discount factor for rewards
    ENTROPY_BETA = 0.01         # entropy regurarlization constant
    USE_GPU = True              # To use GPU, set True
    manual_device = None

    def get_device(self):
        if not self.manual_device:
            return self.manual_device
        if self.USE_GPU:
            return '/gpu:0'
        return '/cpu:0'

    device = property(get_device)

    # TODO: singleton object.

VIEW_CFG = [(30.0, 12), (-30.0, 12), (0, 4), (90, 1), (-90, 1)]
VIEW_CFG_REV2 = [(45.0, 8), (-45.0, 8), (0, 4), (90, 1), (-90, 1)]
VIEW_CFG_REV4 = [(45.0, 4, 45.0), (-45.0, 4, 45.0), (0, 4), (90, 1), (-90, 1)]
#VIEW_CFG_REV4 = [(30.0, 1)]
SV_VISCFG = [ { 'ch_out' : 32, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 32, 'strides' : [1,3,3,1], 'kernel_size' : [5,5]},
        { 'ch_out' : 64, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 128, 'strides' : None, 'kernel_size' : None} ]

SV_SHARED = [ { 'ch_out' : 32, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : None, 'kernel_size' : None}]

SV_NON_SHARED = [ { 'ch_out' : 128, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 256, 'strides' : [1,3,3,1], 'kernel_size' : [5,5]}]

MV_VISCFG = [ { 'ch_out' : 128, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 256, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 256, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 512, 'strides' : None, 'kernel_size' : None} ]

MV_VISCFG2 = [
        { 'ch_out' : 128, 'strides' : [1,3,3,1], 'kernel_size' : [5,5]},
        { 'ch_out' : 256, 'strides' : [1,3,3,1], 'kernel_size' : [5,5]},
        { 'ch_out' : 256, 'strides' : [1,3,3,1], 'kernel_size' : [5,5]},
        { 'ch_out' : 512, 'strides' : [1,3,3,1], 'kernel_size' : [5,5]} ]

SV_NAIVE =  [ { 'ch_out' : 32, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 128, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 256, 'strides' : None, 'kernel_size' : [5,5]},
        { 'ch_out' : 512, 'strides' : None, 'kernel_size' : [5,5]},
        ]

SV_NAIVE_224 =  [ { 'ch_out' : 32, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 128, 'strides' : None, 'kernel_size' : None},
        { 'ch_out' : 256, 'strides' : None, 'kernel_size' : [5,5]},
        { 'ch_out' : 512, 'strides' : None, 'kernel_size' : [5,5]},
        { 'ch_out' : 512, 'strides' : None, 'kernel_size' : [5,5]},
        ]

# VGG16 variant: use strides instead of max polling
SV_VGG16_STRIDES = [
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : [1,3,3,1], 'kernel_size' : None},
        { 'ch_out' : 128, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 128, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 128, 'strides' : [1,3,3,1], 'kernel_size' : None},
        { 'ch_out' : 256, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 256, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 256, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 256, 'strides' : [1,3,3,1], 'kernel_size' : None},
        { 'ch_out' : 512, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 512, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 512, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 512, 'strides' : [1,3,3,1], 'kernel_size' : None},
        { 'ch_out' : 512, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 512, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 512, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 512, 'strides' : [1,3,3,1], 'kernel_size' : None},
        ]

SV_RESNET18 =  [
        # conv1 in the paper, with max pooling moved from conv2
        { 'ch_out' : 32, 'strides' : None, 'kernel_size' : [7,7], 'max_pool': {'kernel_size':[3,3], 'strides':[2,2] } },
        # conv2, note: default kernel_size is 3x3
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'fork'},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'join'},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'fork'},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'join'},
        # conv3
        { 'ch_out' : 128, 'strides' : None, 'kernel_size' : None, 'res': 'fork'},
        { 'ch_out' : 128, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'join'},
        { 'ch_out' : 128, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'fork'},
        { 'ch_out' : 128, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'join'},
        # conv4
        { 'ch_out' : 256, 'strides' : None, 'kernel_size' : None, 'res': 'fork'},
        { 'ch_out' : 256, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'join'},
        { 'ch_out' : 256, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'fork'},
        { 'ch_out' : 256, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'join'},
        # conv5
        { 'ch_out' : 512, 'strides' : None, 'kernel_size' : None, 'res': 'fork'},
        { 'ch_out' : 512, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'join'},
        { 'ch_out' : 512, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'fork'},
        { 'ch_out' : 512, 'strides' : [1,1,1,1], 'kernel_size' : None, 'res': 'join'},
        ]

SV_HOLE_LOWRES = [
        { 'ch_out' : 32, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 32, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 32, 'strides' : [1,1,1,1], 'kernel_size' : None, 'hole' : 1},
        { 'ch_out' : 64, 'strides' : [1,2,2,1], 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None, 'hole' : 1},
        { 'ch_out' : 128, 'strides' : [1,2,2,1], 'kernel_size' : None},
        { 'ch_out' : 128, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 128, 'strides' : [1,1,1,1], 'kernel_size' : None, 'hole' : 1},
        { 'ch_out' : 256, 'strides' : [1,2,2,1], 'kernel_size' : None},
        { 'ch_out' : 256, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 256, 'strides' : [1,1,1,1], 'kernel_size' : None, 'hole' : 1},
        { 'ch_out' : 512, 'strides' : [1,2,2,1], 'kernel_size' : None},
        ]

SV_HOLE_MIDRES = [
        { 'ch_out' : 32, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 32, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 32, 'strides' : [1,1,1,1], 'kernel_size' : None, 'hole' : 1},
        { 'ch_out' : 64, 'strides' : [1,2,2,1], 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None, 'hole' : 1},
        { 'ch_out' : 128, 'strides' : [1,2,2,1], 'kernel_size' : None},
        { 'ch_out' : 128, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 128, 'strides' : [1,1,1,1], 'kernel_size' : None, 'hole' : 1},
        { 'ch_out' : 256, 'strides' : [1,2,2,1], 'kernel_size' : None},
        ]

SV_HOLE_HIGHRES = [
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None, 'hole' : 1},
        { 'ch_out' : 64, 'strides' : [1,2,2,1], 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None},
        { 'ch_out' : 64, 'strides' : [1,1,1,1], 'kernel_size' : None, 'hole' : 1},
        { 'ch_out' : 64, 'strides' : [1,2,2,1], 'kernel_size' : None},
        ]

MAX_STEPS = 1000000
#DEFAULT_RES = 224
DEFAULT_RES = 112
BATCH_SIZE = 32
NUM_TO_EVALUATE = 10000

MAGNITUDES = np.array([0.0125, 0.025], dtype=np.float32)
# STATE_CHECK_DELTAS = np.array([0.00125, 0.0025], dtype=np.float32)
MAX_ITERATION_PER_EPOCH = 16
A3C_LOCAL_T = 32
RL_GAMMA = 0.999 # discount factor for rewards
ENTROPY_BETA = 0.01

STATE_TRANSITION_DEFAULT_MAGS = np.array([0.02, 0.125], dtype=np.float32)
STATE_TRANSITION_DEFAULT_DELTAS = np.array([0.0025, 0.0125], dtype=np.float32)

INVERSE_MODEL_HIDDEN_LAYER = [256]
FORWARD_MODEL_HIDDEN_LAYERS = [256, 288]
POLICY_HIDDEN_LAYERS = [256, 256]
VALUE_HIDDEN_LAYERS = [256, 256]
