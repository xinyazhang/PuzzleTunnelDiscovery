# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
import numpy as np

# env_fn = '../res/alpha/alpha_env-1.2.org.obj'
# rob_fn = '../res/alpha/alpha-1.2.org.obj'
# keys_fn = '../res/alpha/alpha-1.2.org.path'
# keys_w_last = True
# keys_fn = 'blend.path'
# keys_w_last = False

# NOTE: DO NOT LOAD OBJ WITH UV, OTHERWISE IT CREATES DUPLICATED VERTICES
# keys_w_first_npz = '../res/alpha/alpha-1.2.org.w-first.npz'
env_wt_fn = '../res/alpha/alpha_env-1.0.wt.obj'
rob_wt_fn = '../res/alpha/alpha-1.0.wt.obj'
env_uv_fn = '../res/alpha/alpha_env-1.0.wt.tcp.obj'
rob_uv_fn = '../res/alpha/alpha-1.0.wt.tcp.obj'
rob_ompl_center = None
tunnel_v_fn = None # Testing data, no tunnel vertex file
