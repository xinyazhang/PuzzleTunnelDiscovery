# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import numpy as np

env_fn = '../res/alpha/alpha_env-1.2.org.obj'
rob_fn = '../res/alpha/alpha-1.2.org.obj'
keys_fn = '../res/alpha/alpha-1.2.org.path'
keys_w_last = True
# keys_fn = 'blend.path'
# keys_w_last = False

keys_w_first_npz = '../res/alpha/alpha-1.2.org.w-first.npz'
env_wt_fn = '../res/alpha/alpha_env-1.2.wt.obj'
rob_wt_fn = '../res/alpha/alpha-1.2.wt.obj'
rob_ompl_center = np.array([16.973146438598633, 1.2278236150741577, 10.204807281494141])
