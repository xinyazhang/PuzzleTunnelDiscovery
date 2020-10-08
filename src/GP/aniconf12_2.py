# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
import numpy as np

# env_fn = '../res/alpha/alpha_env-1.2.org.obj'
# rob_fn = '../res/alpha/alpha-1.2.org.obj'
keys_fn = '../res/alpha/alpha-1.2.org.path'
keys_w_last = True
# keys_fn = 'blend.path'
# keys_w_last = False

# NOTE: DO NOT LOAD OBJ WITH UV, OTHERWISE IT CREATES DUPLICATED VERTICES
keys_w_first_npz = '../res/alpha/alpha-1.2.org.w-first.npz'
env_wt_fn = '../res/alpha/alpha_env-1.2.wt2.obj'
rob_wt_fn = '../res/alpha/alpha-1.2.wt2.obj'
env_uv_fn = '../res/alpha/alpha_env-1.2.wt2.tcp.obj'
rob_uv_fn = '../res/alpha/alpha-1.2.wt2.tcp.obj'
rob_ompl_center = np.array([16.973146438598633, 1.2278236150741577, 10.204807281494141])
tunnel_v_fn = '../res/alpha/alpha-1.2.org.tunnel.npz'
