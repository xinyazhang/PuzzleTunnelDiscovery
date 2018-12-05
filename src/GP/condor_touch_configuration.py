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
from scipy.misc import imsave

def usage():
    print('''Usage:
1. condor_touch_configuration.py show
    Show the number of tunnel vertices
2. condor_touch_configuration.py run <Batch ID> <Batch Size> <Output Dir>
    Shoot <Batch Size> rays in configuration space originated from <Vertex ID>, and
    store the first collision configurations as one `<Batch ID>.npz` file in Output Dir.
    <Vertex ID> is defined as <Batch ID> mod <Total number of tunnel vertices>.
3. condor_touch_configuration.py isect <Task ID> <Geo Batch Size> <Touch Batch Size> <In/Output Dir>
    Take the output configuration from Step 2) and calculate the intersecting geomery
3. condor_touch_configuration.py project <Task ID> <Geo Batch Size> <Touch Batch Size> <Input Dir>
    [DEBUG] Print the rendered ground truth to out.png
''')

def _create_uw(cmd):
    if cmd == 'project':
        pyosr.init()
        dpy = pyosr.create_display()
        glctx = pyosr.create_gl_context(dpy)
        r = pyosr.Renderer() # 'project' command requires a Renderer
        r.setup()
        # fb = r.render_barycentric(r.BARY_RENDERING_ROBOT, np.array([1024, 1024], dtype=np.int32))
        # imsave('1.png', fb)
        # sys.exit(0)
        r.loadModelFromFile(aniconf.env_uv_fn)
        r.loadRobotFromFile(aniconf.rob_uv_fn)
    else:
        r = pyosr.UnitWorld() # pyosr.Renderer is not avaliable in HTCondor
        r.loadModelFromFile(aniconf.env_wt_fn)
        r.loadRobotFromFile(aniconf.rob_wt_fn)
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

def _fn_touch_q(out_dir, vert_id, batch_id):
    return "{}/touchq-{}-{}.npz".format(out_dir, vert_id, batch_id)

def _fn_isectgeo(out_dir, vert_id, conf_id):
    return "{}/isectgeo-from-vert-{}-{}.obj".format(out_dir, vert_id, conf_id)

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
    assert cmd in ['run', 'isect', 'project'], 'Unknown command {}'.format(cmd)
    uw = _create_uw(cmd)

    if cmd == 'run':
        task_id = int(sys.argv[2])
        batch_id, vert_id = divmod(task_id, len(tunnel_v))
        vertex = tunnel_v[vert_id]
        batch_size = int(sys.argv[3])
        out_dir = sys.argv[4]
        out_fn = _fn_touch_q(out_dir=out_dir, vert_id=vert_id, batch_id=batch_id)

        free_vertices, touch_vertices, to_inf, free_tau, touch_tau = calc_touch(uw, vertex, batch_size)
        np.savez(out_fn,
                 FROM_V=np.repeat(np.array([vertex]), batch_size, axis=0),
                 FREE_V=free_vertices,
                 TOUCH_V=touch_vertices,
                 IS_INF=to_inf,
                 FREE_TAU=free_tau,
                 TOUCH_TAU=touch_tau)
    elif cmd == 'isect':
        task_id = int(sys.argv[2])
        geo_batch_size = int(sys.argv[3])
        tq_batch_size = int(sys.argv[4])
        io_dir = sys.argv[5]
        assert tq_batch_size % geo_batch_size == 0, "Geo Batch Size % Touch Batch Size must be 0"
        '''
        Task partition
        |------------------TQ Batch for Conf. Q--------------------|
        |--Geo Batch--||--Geo Batch--||--Geo Batch--||--Geo Batch--|
        Hence run's task id = isect's task id / (Touch Batch Size/Geo Batch Size)
        '''
        batch_per_tq = tq_batch_size // geo_batch_size
        run_task_id, geo_batch_id = divmod(task_id, batch_per_tq)
        tq_batch_id, vert_id = divmod(run_task_id, len(tunnel_v))
        tq_fn = _fn_touch_q(out_dir=io_dir, vert_id=vert_id, batch_id=tq_batch_id)
        d = np.load(tq_fn)
        tq = d['TOUCH_V']
        is_inf = d['IS_INF']
        for i in range(geo_batch_size):
            per_batch_conf_id = i + geo_batch_id * geo_batch_size
            per_vertex_conf_id = per_batch_conf_id + tq_batch_id * tq_batch_size
            if is_inf[per_batch_conf_id]:
                continue # Skip collding free cases
            V, F = uw.intersecting_geometry(tq[per_batch_conf_id], True)
            pyosr.save_obj_1(V, F, _fn_isectgeo(out_dir=io_dir, vert_id=vert_id, conf_id=per_vertex_conf_id))
    elif cmd == 'project':
        task_id = int(sys.argv[2])
        geo_batch_size = int(sys.argv[3])
        tq_batch_size = int(sys.argv[4])
        io_dir = sys.argv[5]
        assert tq_batch_size % geo_batch_size == 0, "Geo Batch Size % Touch Batch Size must be 0"
        batch_per_tq = tq_batch_size // geo_batch_size
        run_task_id, geo_batch_id = divmod(task_id, batch_per_tq)
        tq_batch_id, vert_id = divmod(run_task_id, len(tunnel_v))
        tq_fn = _fn_touch_q(out_dir=io_dir, vert_id=vert_id, batch_id=tq_batch_id)
        d = np.load(tq_fn)
        tq = d['TOUCH_V']
        is_inf = d['IS_INF']
        if False:
            for i in range(geo_batch_size):
                per_batch_conf_id = i + geo_batch_id * geo_batch_size
                per_vertex_conf_id = per_batch_conf_id + tq_batch_id * tq_batch_size
                if is_inf[per_batch_conf_id]:
                    continue # Skip collding free cases
                iobjfn = _fn_isectgeo(out_dir=io_dir, vert_id=vert_id, conf_id=per_vertex_conf_id)
                V, F = pyosr.load_obj_1(iobjfn)
                print("calling intersecting_to_robot_surface for file {} config {}\n".format(iobjfn, tq[per_batch_conf_id]))
                IF, IBV = uw.intersecting_to_robot_surface(tq[per_batch_conf_id], True, V, F)
                #IF, IBV = uw.intersecting_to_model_surface(tq[per_batch_conf_id], True, V, F)
                V1, F1 = uw.get_robot_geometry(tq[per_batch_conf_id], True)
                pyosr.save_obj_1(IBV, IF, 'idata.obj')
                pyosr.save_obj_1(V1, F1, '1.obj')
                uw.add_barycentric(IF, IBV, uw.BARY_RENDERING_ROBOT)
        else:
            IBV, IF = pyosr.load_obj_1('idata.obj')
            uw.add_barycentric(IF, IBV, uw.BARY_RENDERING_ROBOT)
        fb = uw.render_barycentric(uw.BARY_RENDERING_ROBOT, np.array([1024, 1024], dtype=np.int32))
        imsave('1.png', fb)

if __name__ == '__main__':
    main()
