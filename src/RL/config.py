# -*- coding: utf-8 -*-

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
