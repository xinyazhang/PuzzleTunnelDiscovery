#!/usr/bin/env python2

from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())

import pyosr
import numpy as np
import aniconf12_2 as aniconf
import uw_random
import math

def usage():
    print('''Usage:
1. condor_touch_configuration.py show
    Show the number of tunnel vertices
2. condor_touch_configuration.py run <Batch ID> <Batch Size> <Output Dir>
    Shoot <Batch Size> rays in configuration space originated from <Vertex ID>, and
    store the first collision configurations as one `<Batch ID>.npz` file in Output Dir.
    <Vertex ID> is defined as <Batch ID> mod <Total number of tunnel vertices>.''')

def _create_uw():
    r = pyosr.UnitWorld() # pyosr.Renderer is not avaliable in HTCondor
    r.loadModelFromFile(aniconf.env_fn)
    r.loadRobotFromFile(aniconf.rob_fn)
    r.enforceRobotCenter(aniconf.rob_ompl_center)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)

    return r

def calc_touch(uw, vertex, batch_size):
    q0 = uw.translate_to_unit_state(vertex)
    N_RET = 5
    ret_lists = [[] for i in range(N_RET)]
    for i in range(batch_size):
        tr = uw_random.random_on_sphere(1.0)
        aa = uw_random.random_on_sphere(2 * math.pi)
        to = pyosr.apply(q0, tr, aa)
        tups = uw.transit_state_to_with_contact(q0, to, 0.0125 / 8)
        for i in range(N_RET):
            ret_lists[i].append(tups[i])
    rets = [np.array(ret_lists[i]) for i in range(N_RET)]
    for i in range(N_RET):
        print("{} shape {}".format(i, rets[i].shape))
    return rets

def main():
    if len(sys.argv) < 2:
        usage()
        return
    cmd = sys.argv[1]
    if cmd in ['-h', '--help', 'help']:
        usage()
        return
    tunnel_v = np.load(aniconf.tunnel_v_fn)['TUNNEL_V']
    if cmd in ['show']:
        print("# of tunnel vertices is {}".format(len(tunnel_v)))
        return
    assert cmd == 'run'
    task_id = int(sys.argv[2])
    batch_id, vert_id = divmod(task_id, len(tunnel_v))
    vertex = tunnel_v[vert_id]
    batch_size = int(sys.argv[3])
    out_dir = sys.argv[4]

    uw = _create_uw()

    free_vertices, touch_vertices, to_inf, free_tau, touch_tau = calc_touch(uw, vertex, batch_size)
    np.savez("{}/touchq-{}-{}.npz".format(out_dir, vert_id, batch_id),
             FROM_V=np.repeat(np.array([vertex]), batch_size, axis=0),
             FREE_V=free_vertices,
             TOUCH_V=touch_vertices,
             IS_INF=to_inf,
             FREE_TAU=free_tau,
             TOUCH_TAU=touch_tau)

if __name__ == '__main__':
    main()
