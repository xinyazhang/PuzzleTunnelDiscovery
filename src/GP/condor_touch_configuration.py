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
from task_partitioner import *
# Our modules
import texture_format
import atlas
import task_partitioner

#import progressbar

ATLAS_RES = 2048
STEPPING_FOR_TOUCH = 0.0125 / 8
STEPPING_FOR_CONNECTIVITY = STEPPING_FOR_TOUCH * 4

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
4. condor_touch_configuration.py project <Vertex ID> <Input Dir> [Output PNG]
    REMOVED. This was a debugging function
5. condor_touch_configuration.py uvproj <rob/env> <Task ID> <Mini Batch> <Touch Batch> <Input Dir>
    Project intersection results to rob/env surface as vertex tuples and barycentric coordinates.
6. condor_touch_configuration.py uvrender <rob/env> <Vertex ID> <Input Dir>
    Render the uvproj results to numpy arrays and images.
7. condor_touch_configuration.py atlas2prim <Output Dir>
    Generate the chart that maps pixels in ATLAS image back to PRIMitive ID
8. condor_touch_configuration.py sample <Task ID> <Batch Size> <Input/Output Dir>
    Sample from the product of uvrender, and generate the sample in the narrow tunnel
    TODO: Vertex ID can be 'all'
9. condor_touch_configuration.py samvis <Task ID> <Batch Size> <Input/Output Dir> <PRM sample file>
    Calculate the visibility matrix of the samples from 'sample' Command.
10. condor_touch_configuration.py dump <object name> [Arguments depending on object]
    a) samconf <Vertex ID> <Conf ID or Range> <Input Dir> <Output .obj>
''')


def _fn_atlastex(out_dir, geo_type, vert_id, index=None, nw=False):
    nwsuffix = "" if not nw else "-nw"
    if index is None:
        return "{}/tex-{}-from-vert-{}{}.png".format(out_dir, geo_type, vert_id, nwsuffix)
    else:
        return "{}/tex-{}-from-vert-{}-{}{}.png".format(out_dir, geo_type, vert_id, index, nwsuffix)

def _create_uw(cmd):
    if 'render' in cmd or cmd in ['atlas2prim']:
        pyosr.init()
        dpy = pyosr.create_display()
        glctx = pyosr.create_gl_context(dpy)
        r = pyosr.Renderer() # 'project' command requires a Renderer
        if cmd in ['atlas2prim']:
            r.pbufferWidth = ATLAS_RES
            r.pbufferHeight = ATLAS_RES
        r.setup()
        r.views = np.array([[0.0,0.0]], dtype=np.float32)
    else:
        r = pyosr.UnitWorld() # pyosr.Renderer is not avaliable in HTCondor


    if cmd in ['run', 'isect']:
        # fb = r.render_barycentric(r.BARY_RENDERING_ROBOT, np.array([1024, 1024], dtype=np.int32))
        # imsave('1.png', fb)
        # sys.exit(0)
        r.loadModelFromFile(aniconf.env_wt_fn)
        r.loadRobotFromFile(aniconf.rob_wt_fn)
    else: # All the remaining commands need UV coordinates
        r.loadModelFromFile(aniconf.env_uv_fn)
        r.loadRobotFromFile(aniconf.rob_uv_fn)
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
        aa = uw_random.random_within_sphere(2 * math.pi)
        to = pyosr.apply(q0, tr, aa)
        tups = uw.transit_state_to_with_contact(q0, to, STEPPING_FOR_TOUCH)
        for i in range(N_RET):
            ret_lists[i].append(tups[i])
    rets = [np.array(ret_lists[i]) for i in range(N_RET)]
    for i in range(N_RET):
        print("{} shape {}".format(i, rets[i].shape))
    return rets

def _get_tunnel_v():
    return np.load(aniconf.tunnel_v_fn)['TUNNEL_V']

def run(uw, args):
    task_id = int(args[0])
    batch_size = int(args[1])
    out_dir = args[2]
    tp = TaskPartitioner(out_dir, None, batch_size)

    vertex = tp.get_tunnel_vertex(task_id)
    out_fn = tp.get_tq_fn(task_id)

    free_vertices, touch_vertices, to_inf, free_tau, touch_tau = calc_touch(uw, vertex, batch_size)
    np.savez(out_fn,
             FROM_V=np.repeat(np.array([vertex]), batch_size, axis=0),
             FREE_V=free_vertices,
             TOUCH_V=touch_vertices,
             IS_INF=to_inf,
             FREE_TAU=free_tau,
             TOUCH_TAU=touch_tau)

def isect(uw, args):
    tunnel_v = _get_tunnel_v()
    task_id = int(args[0])
    geo_batch_size = int(args[1])
    tq_batch_size = int(args[2])
    io_dir = args[3]
    tp = TaskPartitioner(io_dir, geo_batch_size, tq_batch_size)
    '''
    Task partition
    |------------------TQ Batch for Conf. Q--------------------|
    |--Geo Batch--||--Geo Batch--||--Geo Batch--||--Geo Batch--|
    Hence run's task id = isect's task id / (Touch Batch Size/Geo Batch Size)
    '''
    batch_per_tq = tq_batch_size // geo_batch_size
    run_task_id, geo_batch_id = divmod(task_id, batch_per_tq)
    tq_batch_id, vert_id = divmod(run_task_id, len(tunnel_v))
    '''
    tq_fn = _fn_touch_q(out_dir=io_dir, vert_id=vert_id, batch_id=tq_batch_id)
    d = np.load(tq_fn)
    tq = d['TOUCH_V']
    is_inf = d['IS_INF']
    '''
    for tq, is_inf, vert_id, conf_id in tp.gen_touch_q(task_id):
        if is_inf:
            continue
        V, F = uw.intersecting_geometry(tq, True)
        pyosr.save_obj_1(V, F, tp.get_isect_fn(vert_id, conf_id))

def uvproj(uw, args):
    geo_type = args[0]
    assert geo_type in ['rob', 'env'], "Unknown geo type {}".format(geo_type)
    task_id = int(args[1])
    gp_batch = int(args[2])
    tq_batch = int(args[3])
    io_dir = args[4]
    tp = TaskPartitioner(io_dir, gp_batch, tq_batch)
    for tq, is_inf, vert_id, conf_id in tp.gen_touch_q(task_id):
        if is_inf:
            continue
        fn = tp.get_isect_fn(vert_id, conf_id)
        V, F = pyosr.load_obj_1(fn)
        if geo_type == 'rob':
            IF, IBV = uw.intersecting_to_robot_surface(tq, True, V, F)
        elif geo_type == 'env':
            IF, IBV = uw.intersecting_to_model_surface(tq, True, V, F)
        else:
            assert False
        fn2 = tp.get_uv_fn(geo_type, vert_id, conf_id)
        print('uvproj of {} to {}'.format(fn, fn2))
        pyosr.save_obj_1(IBV, IF, fn2)

def uvrender(uw, args):
    tunnel_v = _get_tunnel_v()
    uw.avi = False
    geo_type = args[0]
    assert geo_type in ['rob', 'env'], "Unknown geo type {}".format(geo_type)
    if geo_type == 'rob':
        geo_flag = uw.BARY_RENDERING_ROBOT
    elif geo_type == 'env':
        geo_flag = uw.BARY_RENDERING_SCENE
    else:
        assert False
    vert_id = int(args[1])
    io_dir = args[2]
    iq = uw.translate_to_unit_state(tunnel_v[vert_id])
    afb = None
    afb_nw = None

    tq_gen = TouchQGenerator(in_dir=io_dir, vert_id=vert_id)
    obj_gen = UVObjGenerator(in_dir=io_dir, geo_type=geo_type, vert_id=vert_id)
    i = 0
    for tq, is_inf in tq_gen:
        # print('tq {} is_inf {}'.format(tq, is_inf))
        IBV, IF = next(obj_gen) # Must, otherwise does not pair
        if is_inf:
            continue
        if IBV is None or IF is None:
            print('IBV {}'.format(None))
            continue
        print('IBV {}'.format(IBV.shape))
        uw.clear_barycentric(geo_flag)
        uw.add_barycentric(IF, IBV, geo_flag)
        fb = uw.render_barycentric(geo_flag, np.array([ATLAS_RES, ATLAS_RES], dtype=np.int32))
        np.clip(fb, 0, 1, out=fb) # Clip to binary
        nw = texture_format.framebuffer_to_file(fb.astype(np.float32))
        w = nw * (1.0 / np.clip(pyosr.distance(tq, iq), 1e-4, None))
        if afb is None:
            afb = w
            afb_nw = nw
        else:
            afb += w
            afb_nw += nw
            np.clip(afb_nw, 0, 1.0, out=afb_nw) # afb_nw is supposed to be binary
        '''
        print('afb shape {}'.format(afb.shape))
        rgb = np.zeros(list(afb.shape) + [3])
        rgb[...,1] = w
        imsave(_fn_atlastex(io_dir, geo_type, vert_id, i), rgb)
        np.savez(_fn_atlas(io_dir, geo_type, vert_id, i), w)
        if i == 4:
            V1, F1 = uw.get_robot_geometry(tq, True)
            pyosr.save_obj_1(V1, F1, '1.obj')
        if i >= 4:
            break
        if i == 0:
            print('NW NZ {}'.format(nw[np.nonzero(nw)]))
        if i >= 128:
            break
        '''
        i+=1
    rgb = np.zeros(list(afb.shape) + [3])
    rgb[...,1] = afb
    # FIXME: Give savez an explicity array name
    imsave(_fn_atlastex(io_dir, geo_type, vert_id, None), rgb)
    np.savez(_fn_atlas(io_dir, geo_type, vert_id, None), afb)
    rgb[...,1] = afb_nw
    imsave(_fn_atlastex(io_dir, geo_type, vert_id, None, nw=True), rgb)
    np.savez(_fn_atlas(io_dir, geo_type, vert_id, None, nw=True), afb)

def atlas2prim(uw, args):
    r = uw
    r.uv_feedback = True
    r.avi = False
    io_dir = args[0]
    for geo_type,flags in zip(['rob', 'env'], [pyosr.Renderer.NO_SCENE_RENDERING, pyosr.Renderer.NO_ROBOT_RENDERING]):
        r.render_mvrgbd(pyosr.Renderer.UV_MAPPINNG_RENDERING|flags)
        atlas2prim = np.copy(r.mvpid.reshape((r.pbufferWidth, r.pbufferHeight)))
        #imsave(geo_type+'-a2p-nt.png', atlas2prim) # This is for debugging
        atlas2prim = texture_format.framebuffer_to_file(atlas2prim)
        atlas2uv = np.copy(r.mvuv.reshape((r.pbufferWidth, r.pbufferHeight, 2)))
        atlas2uv = texture_format.framebuffer_to_file(atlas2uv)
        np.savez(_fn_atlas2prim(io_dir, geo_type), PRIM=atlas2prim, UV=atlas2uv)
        imsave(geo_type+'-a2p.png', atlas2prim) # This is for debugging

def sample(uw, args):
    task_id = int(args[0])
    batch_size = int(args[1])
    io_dir = args[2]
    tp = TaskPartitioner(io_dir, None, batch_size)
    rob_sampler = atlas.AtlasSampler(tp, 'rob', uw.GEO_ROB, task_id)
    env_sampler = atlas.AtlasSampler(tp, 'env', uw.GEO_ENV, task_id)
    pcloud1 = []
    pn1 = []
    pcloud1x = []
    pn1x = []
    pcloud2 = []
    pn2 = []
    conf = []
    for i in range(batch_size):
        while True:
            tup1 = rob_sampler.sample(uw)
            tup2 = env_sampler.sample(uw)
            q = uw.sample_free_configuration(tup1[0], tup1[1], tup2[0], tup2[1], 1e-6, max_trials=16)
            if uw.is_valid_state(q):
                break
        conf.append(q)
    # print("tqre_fn {}".format(tp.get_tqre_fn(task_id)))
    np.savez(tp.get_tqre_fn(task_id), ReTouchQ=conf)
    return
    #
    # Sanity check code
    #
    fail = 0
    for i in progressbar.progressbar(range(32)):
        while True:
            tup1 = rob_sampler.sample(uw)
            tup2 = env_sampler.sample(uw)
            q = uw.sample_free_configuration(tup1[0], tup1[1], tup2[0], tup2[1], 1e-6, max_trials=16)
            fail += 1
            if uw.is_valid_state(q):
                break
            # print("Reject {}".format(q))
        # print("Accept {}".format(q))
        pcloud1.append(tup1[0])
        pcloud2.append(tup2[0])
        conf.append(q)
        '''
        # Sanity check
        # Test if sample_free_configuration aligns the sample pair
        tup1 = rob_sampler.sample(uw, unit=False)
        tup2 = env_sampler.sample(uw, unit=False)
        pcloud1.append(tup1[0])
        pcloud2.append(tup2[0])
        pn1.append(tup1[1])
        pn2.append(tup2[1])
        q = uw.sample_free_configuration(tup1[0], tup1[1], tup2[0], tup2[1], 1e-6, free_guarantee=False)
        tr,rot = pyosr.decompose_2(q)
        new_point = rot.dot(tup1[0].reshape((3,1)))+tr.reshape((3,1))
        pcloud1x.append(new_point.reshape((3)))
        new_normal = rot.dot(tup1[1].reshape((3,1)))
        pn1x.append(new_normal.reshape((3)))
        conf.append(q)
        '''
    pyosr.save_obj_1(pcloud1, [], 'pc1.obj')
    pyosr.save_obj_1(pcloud2, [], 'pc2.obj')
    np.savez('cf1.npz', Q=conf)
    print("Success ratio {} out of {} = {}".format(len(conf), fail, len(conf) / float(fail)))
    '''
    # pyosr.save_obj_2(V=pcloud1, F=[], CN=pn1, FN=[], TC=[], FTC=[], filename='pc1x.obj')
    # pyosr.save_obj_2(V=pcloud2, F=[], CN=pn2, FN=[], TC=[], FTC=[], filename='pc2x.obj')
    pyosr.save_ply_2(V=pcloud1, F=[], N=pn1, UV=[], filename='pc1x.ply')
    pyosr.save_ply_2(V=pcloud1x, F=[], N=pn1x, UV=[], filename='pc1xx.ply')
    pyosr.save_ply_2(V=pcloud2, F=[], N=pn2, UV=[], filename='pc2x.ply')
    np.savetxt('cf1.txt', conf, fmt='%.17g')
    '''

def samvis(uw, args):
    task_id = int(args[0])
    batch_size = int(args[1])
    io_dir = args[2]
    prm_data = args[3]
    tp = TaskPartitioner(io_dir, None, batch_size)

    mc_reference = np.load(prm_data)['V']
    V = np.load(tp.get_tqre_fn(task_id))['ReTouchQ']
    VM = uw.calculate_visibility_matrix2(V[0:batch_size], True,
                                         mc_reference[0:-1], False,
                                         # mc_reference[0:4], False,
                                         STEPPING_FOR_CONNECTIVITY)
    np.savez(tp.get_tqrevis_fn(task_id), VM=VM, Q=V, VMS=np.sum(VM, axis=-1))

def dump(uw, args):
    target = args[0]
    def _range2list(x):
        result = []
        for part in x.split(','):
            if '-' in part:
                a, b = part.split('-')
                a, b = int(a), int(b)
                result.extend(range(a, b + 1))
            else:
                a = int(part)
                result.append(a)
        return result
    if target == 'samconf':
        vert_id = int(args[1])
        conf_sel = np.array(_range2list(args[2]), dtype=np.int)
        print("Printing {}".format(conf_sel))
        io_dir = args[3]
        out_obj_dir = args[4]
        # Iterate through all ReTouchQ files
        batch_id = 0
        conf_base = 0
        largest = np.max(conf_sel)
        while conf_base < largest:
            fn = task_partitioner.tqre_fn(io_dir, vert_id=vert_id, batch_id=batch_id)
            if not os.path.exists(fn):
                print("Fatal: Cannot find file {}".format(fn))
                return
            Q = np.load(fn)['ReTouchQ']
            indices = np.where(np.logical_and(conf_sel >= conf_base, conf_sel < conf_base + len(Q)))
            indices = indices[0]
            # print("Indices {}".format(indices))
            for index in indices:
                conf_id = conf_sel[index]
                # print("conf_sel index {}".format(index))
                # print("conf_id {}".format(conf_id))
                rem = conf_id - conf_base
                # print("rem {}".format(rem))
                q = Q[rem]
                V1,F1 = uw.get_robot_geometry(q, True)
                V2,F2 = uw.get_scene_geometry(q, True)
                Va = np.concatenate((V1, V2), axis=0)
                F2 += V1.shape[0]
                Fa = np.concatenate((F1, F2), axis=0)
                out_obj = "{}/vert-{}-conf-{}.obj".format(out_obj_dir, vert_id, conf_id)
                # print("Dumping to {}".format(out_obj))
                pyosr.save_obj_1(Va, Fa, out_obj)
            batch_id += 1
            conf_base += len(Q)
    else:
        assert False, "Unknown target "+target

def main():
    if len(sys.argv) < 2:
        usage()
        return
    cmd = sys.argv[1]
    if cmd in ['-h', '--help', 'help']:
        usage()
        return
    if cmd in ['show']:
        print("# of tunnel vertices is {}".format(len(_get_tunnel_v())))
        return
    #assert cmd in ['run', 'isect', 'project', 'uvproj', 'uvrender', 'atlas2prim', 'sample', 'samvis'], 'Unknown command {}'.format(cmd)
    cmdmap = {
            'run' : run,
            'isect' : isect,
            'uvproj' : uvproj,
            'uvrender' : uvrender,
            'atlas2prim' : atlas2prim,
            'sample' : sample,
            'samvis' : samvis,
            'dump' : dump
    }
    uw = _create_uw(cmd)
    cmdmap[cmd](uw, sys.argv[2:])


if __name__ == '__main__':
    main()

def __deprecated():
    assert False, "deprecated"
    vert_id = int(sys.argv[2])
    io_dir = sys.argv[3]
    png_fn = sys.argv[4]
    per_vertex_conf_id = 0
    obj_gen = ObjGenerator(in_dir=io_dir, vert_id=vert_id)
    tq_gen = TouchQGenerator(in_dir=io_dir, vert_id=vert_id)
    i = 0
    for V,F in obj_gen:
        tq, is_inf = next(tq_gen)
        if is_inf:
            continue
        IF, IBV = uw.intersecting_to_robot_surface(tq, True, V, F)
        uw.add_barycentric(IF, IBV, uw.BARY_RENDERING_ROBOT)
        V1, F1 = uw.get_robot_geometry(tq, True)
        pyosr.save_obj_1(V1, F1, 'ir-verts-rob/ir-vert-{}/{}.obj'.format(vert_id, i))
        pyosr.save_obj_1(IBV, IF, 'ir-verts-rob/ir-vert-{}/bary-{}.obj'.format(vert_id, i))
        # print("IBV\n{}".format(IBV))
        # print("{} finished".format(i))
        #if i > 0:
            #break
        i+=1
    fb = uw.render_barycentric(uw.BARY_RENDERING_ROBOT, np.array([ATLAS_RES, ATLAS_RES], dtype=np.int32))
    imsave(png_fn, np.transpose(fb))
    '''
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
    '''
