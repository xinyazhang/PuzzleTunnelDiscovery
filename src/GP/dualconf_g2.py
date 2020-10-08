# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
import numpy as np

#env_fn = '../res/dual/dual_tiny.dt.obj'
#rob_fn = '../res/dual/knotted_ring.dt.obj'
env_fn = '../res/dual/dual-g2.dt.tcp.obj'
rob_fn = '../res/dual/knotted_ring.dt.tcp.obj'
keys_fn = None
keys_w_last = True
# keys_fn = 'blend.path'
# keys_w_last = False

env_wt_fn = env_fn
rob_wt_fn = rob_fn
env_uv_fn = env_fn
rob_uv_fn = rob_fn
rob_ompl_center = None
tunnel_v_fn = None
