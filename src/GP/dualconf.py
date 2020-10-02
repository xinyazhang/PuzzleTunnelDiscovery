# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import numpy as np

#env_fn = '../res/dual/dual_tiny.dt.obj'
#rob_fn = '../res/dual/knotted_ring.dt.obj'
env_fn = '../res/dual/dual.dt.tcp.obj'
rob_fn = '../res/dual/knotted_ring.dt.tcp.obj'
keys_fn = None # Testing data has no path
keys_w_last = True

env_wt_fn = env_fn
rob_wt_fn = rob_fn
env_uv_fn = env_fn
rob_uv_fn = rob_fn
rob_ompl_center = None
tunnel_v_fn = None # Testing data has no tunnel V
