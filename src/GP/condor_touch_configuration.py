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
# Our modules
import texture_format
import atlas

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
    This is a debugging function
5. condor_touch_configuration.py uvproj <rob/env> <Task ID> <Mini Batch> <Touch Batch> <Input Dir>
    Project intersection results to rob/env surface as vertex tuples and barycentric coordinates.
6. condor_touch_configuration.py uvrender <rob/env> <Vertex ID> <Input Dir>
    Render the uvproj results to numpy arrays and images.
7. condor_touch_configuration.py atlas2prim <Output Dir>
    Generate the chart that maps pixels in ATLAS image back to PRIMitive ID
8. condor_touch_configuration.py sample <Task ID> <Batch Size> <Input/Output Dir>
    Sample from the product of uvrender, and generate the sample in the narrow tunnel
    TODO: Vertex ID can be 'all'
8. condor_touch_configuration.py samvis <Task ID> <Batch Size> <Input/Output Dir> <PRM sample file>
    Calculate the visibility matrix of the samples from 'sample' Command.
''')

def _fn_touch_q(out_dir, vert_id, batch_id):
    return "{}/touchq-{}-{}.npz".format(out_dir, vert_id, batch_id)

def _fn_isectgeo(out_dir, vert_id, conf_id):
    return "{}/isectgeo-from-vert-{}-{}.obj".format(out_dir, vert_id, conf_id)

def _fn_uvgeo(out_dir, geo_type, vert_id, conf_id):
    return "{}/{}-uv-from-vert-{}-{}.obj".format(out_dir, geo_type, vert_id, conf_id)

def _fn_atlastex(out_dir, geo_type, vert_id, index=None, nw=False):
    nwsuffix = "" if not nw else "-nw"
    if index is None:
        return "{}/tex-{}-from-vert-{}{}.png".format(out_dir, geo_type, vert_id, nwsuffix)
    else:
        return "{}/tex-{}-from-vert-{}-{}{}.png".format(out_dir, geo_type, vert_id, index, nwsuffix)

def _fn_atlas(out_dir, geo_type, vert_id, index=None, nw=False):
    nwsuffix = "" if not nw else "-nw"
    if index is None:
        return "{}/atlas-{}-from-vert-{}{}.npz".format(out_dir, geo_type, vert_id, nwsuffix)
    else:
        return "{}/atlas-{}-from-vert-{}-{}{}.npz".format(out_dir, geo_type, vert_id, index, nwsuffix)

def _fn_atlas2prim(out_dir, geo_type):
    return "{}/atlas2prim-{}.npz".format(out_dir, geo_type)

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

class TaskPartitioner(object):
    '''
    iodir: input/output directory. all task files should be there.
    gp_batch: size of geometry processing batch. Task granularity of `isectgeo` and `uv`
    tq_batch: size of touch configuration batch. Task granularity of `run`
              Note geometry processing needs the touch configuration info.
    '''
    def __init__(self, iodir, gp_batch, tq_batch):
        self._iodir = iodir
        if gp_batch is not None:
            assert tq_batch % gp_batch == 0, "GeoP Batch Size % Touch Batch Size must be 0"
            '''
            Batch subdivider, geometry processing is consider more expensive than tq sampling
            '''
            self._gp_per_tq = tq_batch / gp_batch
        self._gp_batch = gp_batch
        self._tq_batch = tq_batch
        self.tunnel_v = np.load(aniconf.tunnel_v_fn)['TUNNEL_V']

    '''
    Task vector is resized into (batch_id, vertex_id) matrix
    '''
    def get_batch_vert_index(self, task_id):
        return divmod(task_id, len(self.tunnel_v))

    def get_vert_id(self, task_id):
        return self.get_batch_vert_index(task_id)[1]

    def get_batch_id(self, task_id):
        return self.get_batch_vert_index(task_id)[0]

    def get_tunnel_vertex(self, task_id):
        return self.tunnel_v[self.get_vert_id(task_id)]

    def get_tq_batch_size(self):
        return self._tq_batch

    def _task_id_gp_to_tq(self, task_id):
        return divmod(task_id, self._gp_per_tq)

    '''
    Functions gen_touch_q
    Return a generator that "pumps" touch along with its attributes from a given (GeoP) task id
    '''
    def gen_touch_q(self, task_id, members=['TOUCH_V', 'IS_INF']):
        def tqgen(npd, start, size, vert_id, conf_id_base, members):
            sample_array = [d[n] for n in members]
            for i in range(size):
                vc = [vert_id, conf_id_base + start + i]
                sample = [array[start+i] for array in sample_array]
                yield sample + vc
        tq_task_id, remainder = self._task_id_gp_to_tq(task_id)
        d = np.load(self.get_tq_fn(tq_task_id))
        return tqgen(d,
                remainder * self._gp_batch, self._gp_batch,
                self.get_vert_id(tq_task_id),
                self.get_batch_id(tq_task_id) * self._tq_batch,
                members=members)

    '''
    Functions to get I/O file name
    Note: we use (vertex id, configuration id) to uniquely locate a file for geometry processing.
          This tuple is generated by the generator returned from gen_touch_q
    '''
    def get_tq_fn(self, task_id):
        batch_id, vert_id = self.get_batch_vert_index(task_id)
        return _fn_touch_q(out_dir=self._iodir, vert_id=vert_id, batch_id=batch_id)

    def get_isect_fn(self, vert_id, conf_id):
        return _fn_isectgeo(out_dir=self._iodir, vert_id=vert_id, conf_id=conf_id)

    def get_uv_fn(self, geo_type, vert_id, conf_id):
        return _fn_uvgeo(self._iodir, geo_type, vert_id, conf_id)

    def get_atlas_fn(self, geo_type, task_id):
        batch_id, vert_id = self.get_batch_vert_index(task_id)
        return _fn_atlas(self._iodir, geo_type, vert_id, None)

    def get_atlas2prim_fn(self, geo_type):
        return _fn_atlas2prim(self._iodir, geo_type)

    def get_tqre_fn(self, task_id):
        batch_id, vert_id = self.get_batch_vert_index(task_id)
        return "{}/touchq_re-from-vert-{}-{}.npz".format(self._iodir, vert_id, batch_id)

    def get_tqrevis_fn(self, task_id):
        batch_id, vert_id = self.get_batch_vert_index(task_id)
        return "{}/touchq_re_vis-from-vert-{}-{}.npz".format(self._iodir, vert_id, batch_id)

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

class ObjGenerator(object):
    def __init__(self, in_dir, vert_id):
        self.in_dir = in_dir
        self.vert_id = vert_id
        self.per_vertex_conf_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        fn = _fn_isectgeo(out_dir=self.in_dir,
                          vert_id=self.vert_id,
                          conf_id=self.per_vertex_conf_id)
        if not os.path.exists(fn):
            raise StopIteration
        print("loading {}".format(fn))
        self.per_vertex_conf_id += 1
        return pyosr.load_obj_1(fn)

    # Python 2 compat
    def next(self):
        return self.__next__()

class UVObjGenerator(object):
    def __init__(self, in_dir, geo_type, vert_id):
        self.in_dir = in_dir
        self.geo_type = geo_type
        self.vert_id = vert_id
        self.conf_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        fn = _fn_uvgeo(self.in_dir, self.geo_type, self.vert_id, self.conf_id)
        self.conf_id += 1
        print("loading {}".format(fn))
        if not os.path.exists(fn):
            return None, None # Note: do NOT raise StopIteration, we may miss some file in the middle
        return pyosr.load_obj_1(fn)

    # Python 2 compat
    def next(self):
        return self.__next__()


class TouchQGenerator(object):
    def __init__(self, in_dir, vert_id):
        self.in_dir = in_dir
        self.vert_id = vert_id
        self.tq_batch_id = 0
        self.tq_local_id = 0
        self.tq = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.tq is None:
            tq_fn = _fn_touch_q(out_dir=self.in_dir,
                                vert_id=self.vert_id,
                                batch_id=self.tq_batch_id)
            try:
                print("loading {}".format(tq_fn))
                d = np.load(tq_fn)
                self.tq = d['TOUCH_V']
                self.tq_size = len(self.tq)
                self.inf = d['IS_INF']
                self.tq_local_id = 0
            except IOError:
                raise StopIteration
        ret = (self.tq[self.tq_local_id], self.inf[self.tq_local_id])
        self.tq_local_id += 1
        if self.tq_local_id >= self.tq_size:
            self.tq_batch_id += 1
            self.tq_local_id = 0
            self.tq = None
        return ret

    # Python 2 compat
    def next(self):
        return self.__next__()

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
    assert cmd in ['run', 'isect', 'project', 'uvproj', 'uvrender', 'atlas2prim', 'sample', 'samvis'], 'Unknown command {}'.format(cmd)
    uw = _create_uw(cmd)


    if cmd == 'run':
        task_id = int(sys.argv[2])
        out_dir = sys.argv[4]
        batch_size = int(sys.argv[3])
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
    elif cmd == 'isect':
        task_id = int(sys.argv[2])
        geo_batch_size = int(sys.argv[3])
        tq_batch_size = int(sys.argv[4])
        io_dir = sys.argv[5]
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
    elif cmd == 'uvproj':
        args = sys.argv[2:]
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
    elif cmd == 'uvrender':
        uw.avi = False
        args = sys.argv[2:]
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
    elif cmd == 'atlas2prim':
        r = uw
        r.uv_feedback = True
        r.avi = False
        io_dir = sys.argv[2]
        for geo_type,flags in zip(['rob', 'env'], [pyosr.Renderer.NO_SCENE_RENDERING, pyosr.Renderer.NO_ROBOT_RENDERING]):
            r.render_mvrgbd(pyosr.Renderer.UV_MAPPINNG_RENDERING|flags)
            atlas2prim = np.copy(r.mvpid.reshape((r.pbufferWidth, r.pbufferHeight)))
            #imsave(geo_type+'-a2p-nt.png', atlas2prim) # This is for debugging
            atlas2prim = texture_format.framebuffer_to_file(atlas2prim)
            atlas2uv = np.copy(r.mvuv.reshape((r.pbufferWidth, r.pbufferHeight, 2)))
            atlas2uv = texture_format.framebuffer_to_file(atlas2uv)
            np.savez(_fn_atlas2prim(io_dir, geo_type), PRIM=atlas2prim, UV=atlas2uv)
            imsave(geo_type+'-a2p.png', atlas2prim) # This is for debugging
    elif cmd == 'sample':
        args = sys.argv[2:]
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
    elif cmd == 'samvis':
        args = sys.argv[2:]
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
    elif cmd == 'project':
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

if __name__ == '__main__':
    main()
