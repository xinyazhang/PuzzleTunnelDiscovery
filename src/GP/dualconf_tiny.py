import numpy as np

env_fn = '../res/dual/dual_tiny.dt.obj'
rob_fn = '../res/dual/knotted_ring.dt.obj'
keys_fn = '../res/dual/dual_tiny.path'
keys_w_last = True
# keys_fn = 'blend.path'
# keys_w_last = False

# NOTE: DO NOT LOAD OBJ WITH UV, OTHERWISE IT CREATES DUPLICATED VERTICES
env_wt_fn = env_fn
rob_wt_fn = rob_fn
env_uv_fn = '../res/dual/dual_tiny.dt.tcp.obj'
rob_uv_fn = '../res/dual/knotted_ring.dt.tcp.obj'
rob_ompl_center = None
tunnel_v_fn = None # Unknown yet, just a placeholder during the preprocessing.
