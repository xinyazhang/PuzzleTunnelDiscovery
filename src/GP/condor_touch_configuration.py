#!/usr/bin/env python2

from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())
import shutil
import argparse
from six.moves import configparser
import importlib

import pyosr
import numpy as np
#import aniconf12_2 as aniconf
#import aniconf10 as aniconf
#import dualconf_tiny as aniconf
#import dualconf_g2 as aniconf
#import dualconf_g4 as aniconf
#import dualconf as aniconf

#import aniconf12_2
#import aniconf10
#import dualconf_tiny
#import dualconf_g2
#import dualconf_g4
#import dualconf

import uw_random
import math
from scipy.misc import imsave, imread
from scipy.io import savemat, loadmat
from task_partitioner import *
# Our modules
import texture_format
import atlas
import task_partitioner
import disjoint_set
try:
    from progressbar import progressbar
except ImportError:
    progressbar = lambda x: x

import aux

ATLAS_RES = 2048
STEPPING_FOR_TOUCH = 0.0125 / 8
STEPPING_FOR_CONNECTIVITY = STEPPING_FOR_TOUCH * 4

######################################################################################################
#            FIXME: USE ARGPARSE AND SUBPARSERS TO HANDLE ARGUMENTS AND DOCUMENTATION                #
######################################################################################################

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
6.a (Auxiliary function) condor_touch_configuration.py uvmerge <Input/Output Dir>
    Take the output of `uvrender` and write the mean of atlas to I/O dir.
6.b (Auxiliary function) condor_touch_configuration.py atlas2prim <Output Dir>
    Generate the chart that maps pixels in ATLAS image back to PRIMitive ID
6.c (Auxiliary function) condor_touch_configuration.py useatlas <rob/env> <npz file of prediction> <Output Dir>
    Copy the prediction from NN to the output dir, with a matching name for `sample` command.
7. condor_touch_configuration.py sample <Task ID> <Batch Size> <Input/Output Dir>
    Sample from the product of uvrender, and generate the sample in the narrow tunnel
    TODO: Vertex ID can be 'all'
7.1. condor_touch_configuration.py sample_enumaxis <Task ID> <Batch Size> <Input/Output Dir>
    Sample from the product of uvrender, enumerate the axis angle w.r.t the contact normal, and only keep the c-free configurations.
8. condor_touch_configuration.py samvis <Task ID> <Batch Size> <Input/Output Dir> <PRM sample file>
    Calculate the visibility matrix of the samples from 'sample' Command.
9. condor_touch_configuration.py dump <object name> [Arguments depending on object]
    a) samconf <Vertex ID> <Conf ID or Range> <Input Dir> <Output dir>
        This
    b) tconf <Vertex ID> <Conf ID or Range> <Input Dir> <Output dir>
        same as samconf, but this takes samples from Command `run`
    c) omplsam <input dir> <output .txt or .npz>
        This dumps the samples to a text file in OMPL form defined by the demo application of RRT sample injection.
        Note: sample injection is our extension rather than an official function.
10. condor_touch_configuration.py samstat <Input/Output Dir> <PRM sample file>
    Parse the output from 'samvis' command, and show the statistics
    Note: we need PRM sample file to know samples used in Monte Carlo visibility algorithm
''')


def _fn_atlastex(out_dir, geo_type, vert_id, index=None, nw=False):
    nwsuffix = "" if not nw else "-nw"
    if index is None:
        return "{}/tex-{}-from-vert-{}{}.png".format(out_dir, geo_type, vert_id, nwsuffix)
    else:
        return "{}/tex-{}-from-vert-{}-{}{}.png".format(out_dir, geo_type, vert_id, index, nwsuffix)

def _create_uw(aniconf, cmd):
    if cmd == 'show':
        return None # show command does not need unit world object
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

    if aniconf.rob_ompl_center is None:
        '''
        When OMPL_center is not specified
        '''
        assert aniconf.env_wt_fn == aniconf.env_uv_fn
        assert aniconf.rob_wt_fn == aniconf.rob_uv_fn

    if cmd in ['run', 'isect']:
        # fb = r.render_barycentric(r.BARY_RENDERING_ROBOT, np.array([1024, 1024], dtype=np.int32))
        # imsave('1.png', fb)
        # sys.exit(0)
        r.loadModelFromFile(aniconf.env_wt_fn)
        r.loadRobotFromFile(aniconf.rob_wt_fn)
    else: # All the remaining commands need UV coordinates
        r.loadModelFromFile(aniconf.env_uv_fn)
        r.loadRobotFromFile(aniconf.rob_uv_fn)
    if aniconf.rob_ompl_center is not None:
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
    if aniconf.tunnel_v_fn is None:
        return None
    return np.load(aniconf.tunnel_v_fn)['TUNNEL_V']

class TouchConfiguration(object):

    def __init__(self, args):
        self._args = args
        self._tv_cache = None
        #self._aniconf = importlib.import_module('puzzleconf.'+args.puzzle)
        self._aniconf = importlib.import_module(args.puzzle)
        self._uw = _create_uw(self._aniconf, args.command)

    def _get_tunnel_v(self):
        aniconf = self._aniconf
        if aniconf.tunnel_v_fn is None:
            return None
        if self._tv_cache is None:
            self._tv_cache = np.load(aniconf.tunnel_v_fn)['TUNNEL_V']
        return self._tv_cache

    @staticmethod
    def _setup_parser_show(subparsers):
        show_parser = subparsers.add_parser("show", help='Show the number of tunnel vertices')

    def show(self):
        print("# of tunnel vertices is {}".format(len(self._get_tunnel_v())))

    @staticmethod
    def _setup_parser_run(subparsers):
        run_parser = subparsers.add_parser("run",
                help='Sample #task_size touch configurations from Tunnel Vertex (#task_id mod (total number of tunnel vertices))')
        run_parser.add_argument('task_id',
                help='Index of the batch to process',
                type=int)
        run_parser.add_argument('batch_size',
                help='Size of the batch to process',
                type=int)
        run_parser.add_argument('out_dir',
                help='Size of the batch to process',
                type=str)

    def run(self):
        task_id = self._args.task_id
        batch_size = self._args.batch_size
        out_dir = self._args.out_dir
        tp = TaskPartitioner(out_dir, None, batch_size, tunnel_v=self._get_tunnel_v())

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

    @staticmethod
    def _setup_parser_isect(subparsers):
        isect_parser = subparsers.add_parser("isect",
                help='Calculate the intersecting geometry from touch configurations')
        isect_parser.add_argument('task_id',
                help='Index of the batch to process',
                type=int)
        isect_parser.add_argument('geo_batch_size',
                help='Task granularity. Must divide tq_batch_size',
                type=int)
        isect_parser.add_argument('tq_batch_size',
                help='Number of samples in touch configuration files. Must be the same from \'run\' command',
                type=int)
        isect_parser.add_argument('io_dir',
                help='Directory of the input samples and output geometries',
                type=str)

    def isect(self):
        uw = self._uw
        tunnel_v = self._get_tunnel_v()
        task_id = self._args.task_id
        geo_batch_size = self._args.geo_batch_size
        tq_batch_size = self._args.tq_batch_size
        io_dir = self._args.io_dir

        tp = TaskPartitioner(io_dir, geo_batch_size, tq_batch_size, tunnel_v=tunnel_v)
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
            np.savez_compressed(tp.get_isect_fn(vert_id, conf_id), V=V, F=F)
            #pyosr.save_obj_1(V, F, tp.get_isect_fn(vert_id, conf_id))

    @staticmethod
    def _setup_parser_uvproj(subparsers):
        uvproj_parser = subparsers.add_parser("uvproj",
                help='Project intersection results to rob/env surface as vertex tuples and barycentric coordinates')
        uvproj_parser.add_argument('geo_type', choices=['rob', 'env'])
        uvproj_parser.add_argument('task_id',
                help='Index of the batch to process', type=int)
        uvproj_parser.add_argument('geo_batch_size',
                help='Task granularity. Must divide tq_batch_size',
                type=int)
        uvproj_parser.add_argument('tq_batch_size',
                help='Number of samples in touch configuration files. Must be the same from \'run\' command',
                type=int)
        uvproj_parser.add_argument('io_dir',
                help='Directory of the input geometries and output projection data',
                type=str)

    def uvproj(self):
        uw = self._uw
        geo_type = self._args.geo_type
        task_id = self._args.task_id
        gp_batch = self._args.geo_batch_size
        tq_batch = self._args.tq_batch_size
        io_dir = self._args.io_dir
        tp = TaskPartitioner(io_dir, gp_batch, tq_batch, tunnel_v=self._get_tunnel_v())
        for tq, is_inf, vert_id, conf_id in tp.gen_touch_q(task_id):
            if is_inf:
                continue
            fn = tp.get_isect_fn(vert_id, conf_id)
            #V, F = pyosr.load_obj_1(fn)
            d = np.load(fn+'.npz')
            V = d['V']
            F = d['F']
            if geo_type == 'rob':
                IF, IBV = uw.intersecting_to_robot_surface(tq, True, V, F)
            elif geo_type == 'env':
                IF, IBV = uw.intersecting_to_model_surface(tq, True, V, F)
            else:
                assert False
            fn2 = tp.get_uv_fn(geo_type, vert_id, conf_id)
            #print('uvproj of {} to {}'.format(fn, fn2))
            #pyosr.save_obj_1(IBV, IF, fn2)
            np.savez_compressed(fn2, V=IBV, F=IF)

    @staticmethod
    def _setup_parser_uvrender(subparsers):
        uvrender_parser = subparsers.add_parser("uvrender",
                help='Render and accumulate results from uvproj to numpy arrays and images.')
        uvrender_parser.add_argument('geo_type', choices=['rob', 'env'])
        uvrender_parser.add_argument('vert_id',
                help='Vertex ID with in the narrow tunnel vertices',
                type=int)
        uvrender_parser.add_argument('io_dir',
                help='Directory of the input projection data and rendered results',
                type=int)


    def uvrender(self):
        uw = self._uw
        tunnel_v = self._get_tunnel_v()
        uw.avi = False
        geo_type = self._args.geo_type
        TYPE_TO_FLAG = {'rob' : uw.BARY_RENDERING_ROBOT,
                        'env' : uw.BARY_RENDERING_SCENE }
        geo_flag = TYPE_TO_FLAG[geo_type]

        vert_id = self._args.vert_id
        io_dir = self._args.io_dir
        iq = uw.translate_to_unit_state(tunnel_v[vert_id])
        afb = None
        afb_nw = None

        tq_gen = TouchQGenerator(in_dir=io_dir, vert_id=vert_id)
        obj_gen = UVObjGenerator(in_dir=io_dir, geo_type=geo_type, vert_id=vert_id)
        i = 0
        DEBUG_UVRENDER = False
        for tq, is_inf in tq_gen:
            # print('tq {} is_inf {}'.format(tq, is_inf))
            IBV, IF = next(obj_gen) # Must, otherwise does not pair
            if is_inf:
                continue
            if IBV is None or IF is None:
                print('IBV {}'.format(None))
                continue
            print('{}: IBV {} IF {}'.format(i, IBV.shape, IF.shape))
            if DEBUG_UVRENDER:
                svg_fn = '{}.svg'.format(i)
                # Paint everything ...
                if i == 0 and geo_type == 'rob':
                    V, F = uw.get_robot_geometry(tq, True)
                    print("V {}\nF {}".format(V.shape, F.shape))
                    IF, IBV = uw.intersecting_to_robot_surface(tq, True, V, F)
                    print("IF {}\nIBV {}\n{}".format(IF.shape, IBV.shape, IBV[:5]))
                    '''
                    NPICK=3000
                    IF = IF[:NPICK]
                    IBV = IBV[:NPICK*3]
                    '''
            else:
                svg_fn = ''
            uw.clear_barycentric(geo_flag)
            uw.add_barycentric(IF, IBV, geo_flag)
            if DEBUG_UVRENDER and i == 2:
                print("BaryF {}".format(IF))
                print("Bary {}".format(IBV))
            fb = uw.render_barycentric(geo_flag,
                                       np.array([ATLAS_RES, ATLAS_RES], dtype=np.int32),
                                       svg_fn=svg_fn)
            #np.clip(fb, 0, 1, out=fb) # Clip to binary
            nw = texture_format.texture_to_file(fb.astype(np.float32))
            distance = np.clip(pyosr.distance(tq, iq), 1e-4, None)
            w = nw * (1.0 / distance)
            if afb is None:
                afb = w
                afb_nw = nw
            else:
                afb += w
                afb_nw += nw
                np.clip(afb_nw, 0, 1.0, out=afb_nw) # afb_nw is supposed to be binary
            # Debugging code
            if DEBUG_UVRENDER:
                print('afb shape {}'.format(afb.shape))
                print('distance {}'.format(distance))
                rgb = np.zeros(list(afb.shape) + [3])
                rgb[...,1] = w
                imsave(_fn_atlastex(io_dir, geo_type, vert_id, i), rgb)
                np.savez(atlas_fn(io_dir, geo_type, vert_id, i), w)
                print('NW NZ {}'.format(nw[np.nonzero(nw)]))
                V1, F1 = uw.get_robot_geometry(tq, True)
                pyosr.save_obj_1(V1, F1, '{}.obj'.format(i))
                V2, F2 = uw.get_scene_geometry(tq, True)
                pyosr.save_obj_1(V2, F2, '{}e.obj'.format(i))
                if i >= 16:
                   return
            i+=1
        rgb = np.zeros(list(afb.shape) + [3])
        rgb[...,1] = afb
        # FIXME: Give savez an explicity array name
        imsave(_fn_atlastex(io_dir, geo_type, vert_id, None), rgb)
        np.savez(atlas_fn(io_dir, geo_type, vert_id, None), afb)
        rgb[...,1] = afb_nw
        imsave(_fn_atlastex(io_dir, geo_type, vert_id, None, nw=True), rgb)
        np.savez(atlas_fn(io_dir, geo_type, vert_id, None, nw=True), afb)

    @staticmethod
    def _setup_parser_uvmerge(subparsers):
        uvmerge_parser = subparsers.add_parser("uvmerge",
                help='Take the output of `uvrender` and write the mean of atlas to I/O dir.')
        uvmerge_parser.add_argument('io_dir', help='Input/Output directory')

    def uvmerge(self):
        io_dir = self._args.io_dir
        import npz_mean
        for geo_type in ['rob', 'env']:
            for nw in [False, True]: # Not Weighted
                fns = []
                for vert_id, _ in enumerate(iter(int, 1)):
                    fn = task_partitioner.atlas_fn(io_dir, geo_type, vert_id, None, nw=nw)
                    if not os.path.exists(fn):
                        break
                    fns.append(fn)
                if nw:
                    suffix = '-nw'
                else:
                    suffix = ''
                mean_fn = '{}/atlas-{}-mean{}.npz'.format(io_dir, geo_type, suffix)
                mean_img = '{}/atlas-{}-mean{}.png'.format(io_dir, geo_type, suffix)
                fns.append(mean_fn)
                print(fns)
                npz_mean.main(fns)
                d = np.load(mean_fn)
                img = d[d.keys()[0]]
                if nw:
                    img = img.astype(bool).astype(float)
                else:
                    print("sum 0 {}".format(np.sum(img)))
                    '''
                    nz = img[np.nonzero(img)]
                    import scipy.stats
                    import matplotlib.pyplot as plt
                    plt.hist(nz, 50, normed=1, facecolor='green', alpha=0.5);
                    plt.show()
                    '''
                    '''
                    # Method 1
                    print("sum 1 {}".format(np.sum(nz)))
                    m = np.mean(nz)
                    #np.clip(img, 0, m, out=img)
                    # img2 = np.clip(img, m, None) # Cut less important points
                    img2 = np.copy(img)
                    img2[img < m] = 0.0
                    nz2 = img2[np.nonzero(img2)]
                    print("sum 2 {}".format(np.sum(nz2)))
                    m2 = np.median(nz2)
                    np.clip(img2, m2, None, img)
                    print("First clip thresh {}. Second clip thresh {}".format(m, m2))
                    '''
                    # Method 2
                    for i in range(2):
                        nzi = np.nonzero(img)
                        print("nz count {} {}".format(i, len(nzi[0])))
                        nz = img[nzi]
                        m = np.mean(nz)
                        img[img < m] = 0.0
                        print("sum {} {}".format(i+1, np.sum(img)))
                    #np.clip(img, 0, np.mean(m), out=img)
                    '''
                    # Method 3
                    m = math.sqrt(np.max(img))
                    img[img < m] = 0.0
                    print("sum 1 {}".format(np.sum(img)))
                    '''
                imsave(mean_img, img)

    @staticmethod
    def _setup_parser_atlas2prim(subparsers):
        atlas2prim_parser = subparsers.add_parser("atlas2prim",
                help='Generate the chart that maps pixels in ATLAS image back to PRIMitive ID.')
        atlas2prim_parser.add_argument('out_dir', help='Output directory')

    def atlas2prim(self):
        r = uw = self._uw
        r.uv_feedback = True
        r.avi = False
        io_dir = self._args.out_dir
        for geo_type,flags in zip(['rob', 'env'], [pyosr.Renderer.NO_SCENE_RENDERING, pyosr.Renderer.NO_ROBOT_RENDERING]):
            r.render_mvrgbd(pyosr.Renderer.UV_MAPPINNG_RENDERING|flags)
            atlas2prim = np.copy(r.mvpid.reshape((r.pbufferWidth, r.pbufferHeight)))
            #imsave(geo_type+'-a2p-nt.png', atlas2prim) # This is for debugging
            atlas2prim = texture_format.framebuffer_to_file(atlas2prim)
            atlas2uv = np.copy(r.mvuv.reshape((r.pbufferWidth, r.pbufferHeight, 2)))
            atlas2uv = texture_format.framebuffer_to_file(atlas2uv)
            np.savez(task_partitioner.atlas2prim_fn(io_dir, geo_type), PRIM=atlas2prim, UV=atlas2uv)
            imsave(geo_type+'-a2p.png', atlas2prim) # This is for debugging

    @staticmethod
    def _setup_parser_useatlas(subparsers):
        useatlas_parser = subparsers.add_parser("useatlas",
                help='Copy the prediction from NN to the output dir, with a matching name for `sample` command.')
        useatlas_parser.add_argument('geo_type', choices=['rob', 'env'])
        useatlas_parser.add_argument('npz',
                help='NPZ/PNG file that stores the prediction')
        useatlas_parser.add_argument('out_dir',
                help='Output directory')
        useatlas_parser.add_argument('--uniform_weight', help='Treat all non-zero weights uniformly (useful for debugging)', action='store_true')

    def useatlas(self):
        geo_type = self._args.geo_type
        fn = self._args.npz
        io_dir = self._args.out_dir
        tp = TaskPartitioner(io_dir, None, None, tunnel_v=self._get_tunnel_v())
        ofn = task_partitioner.atlas_fn(io_dir, geo_type, 0)
        if fn.endswith('.npz'):
            print("input file must be .npz format")
            shutil.copyfile(fn, ofn)
            print("Copied file {} -> {}".format(fn, ofn))
        elif fn.endswith('.png'):
            img = imread(fn)
            if self._args.uniform_weight:
                atlas = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.float32)
                atlas[img[...,1].nonzero()] = 1.0
            else:
                if len(img.shape) == 3:
                    atlas = img[...,1].astype(np.float32)
                else:
                    atlas = img.astype(np.float32)
            np.savez(ofn, ATEX=atlas)
            print(img.shape)
            print(atlas.shape)
            imsave('debug.png', atlas)
            print("Translate file {} -> {}".format(fn, ofn))
        else:
            print("Unknown file extension for {}".format(fn))
            return

    @staticmethod
    def _setup_parser_sample(subparsers):
        common = argparse.ArgumentParser(add_help=False)
        common.add_argument('task_id', help='Task ID', type=int)
        common.add_argument('batch_size', help='Number of samples', type=int)
        common.add_argument('io_dir', help='Input/Output directory')

        sp = subparsers.add_parser('sample',
                help='Sample from the product of uvrender, and generate samples in the narrow tunnel',
                parents=[common])
        spe = subparsers.add_parser('sample_enumaxis',
                help='like \'sample\', but the relative rotation is enumerated rather than sampled',
                parents=[common])
        spe.add_argument('--rotations', help='Number of rotations to enumerate', default=64)

    @staticmethod
    def _setup_parser_sample_enumaxis(subparsers):
        pass # Already done in _setup_parser_sample

    def sample(self):
        self._common_sample(enum_axis=False)

    def sample_enumaxis(self):
        self._common_sample(enum_axis=True)

    def _common_sample(self, enum_axis=False):
        uw = self._uw
        task_id = self._args.task_id
        batch_size = self._args.batch_size
        io_dir = self._args.io_dir
        tp = TaskPartitioner(io_dir, None, batch_size, tunnel_v=self._get_tunnel_v())
        rob_sampler = atlas.AtlasSampler(tp, 'rob', uw.GEO_ROB, task_id)
        env_sampler = atlas.AtlasSampler(tp, 'env', uw.GEO_ENV, task_id)
        pcloud1 = []
        pn1 = []
        pcloud1x = []
        pn1x = []
        pcloud2 = []
        pn2 = []
        conf = []
        signature_conf = []
        SANITY_CHECK=False
        if not SANITY_CHECK:
            for i in progressbar(range(batch_size)):
                if enum_axis:
                    tup1 = rob_sampler.sample(uw)
                    tup2 = env_sampler.sample(uw)
                    qs = uw.enum_free_configuration(tup1[0], tup1[1], tup2[0], tup2[1], 1e-6, denominator=64)
                    for q in qs:
                        conf.append(q)
                        # break # debug: only one sample per touch
                    if len(qs) > 0:
                        pcloud2.append(tup2[0])
                        signature_conf.append(qs[0])
                else:
                    while True:
                        tup1 = rob_sampler.sample(uw)
                        tup2 = env_sampler.sample(uw)
                        q = uw.sample_free_configuration(tup1[0], tup1[1], tup2[0], tup2[1], 1e-6, max_trials=16)
                        if uw.is_valid_state(q):
                            break
                    conf.append(q)
            # print("tqre_fn {}".format(tp.get_tqre_fn(task_id)))
            np.savez(tp.get_tqre_fn(task_id), ReTouchQ=conf, SigReTouchQ=signature_conf)
            np.savetxt('pc2.txt', pcloud2, fmt='%.17g')
        else:
            #
            # Sanity check code
            #
            fail = 0
            for i in progressbar(range(32)):
                while True:
                    tup1 = rob_sampler.sample(uw)
                    tup2 = env_sampler.sample(uw)
                    q = uw.sample_free_configuration(tup1[0], tup1[1], tup2[0], tup2[1], 1e-6, max_trials=16)
                    fail += 1
                    if uw.is_valid_state(q):
                        break
                    print("Reject {}".format(q))
                print("Accept {}".format(q))
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

    @staticmethod
    def _setup_parser_samvis(subparsers):
        sp = subparsers.add_parser('samvis',
                help="Calculate the visibility matrix of the samples from 'sample' Command.")
        sp.add_argument('task_id', help='Task ID', type=int)
        sp.add_argument('batch_size',
                help='Number of sampled touch configurations. -1 to cover all samples',
                type=int)
        sp.add_argument('io_dir', help='Input/Output directory')
        sp.add_argument('prm', help='Sample set for visibility estimation')

    def samvis(self):
        task_id = self._args.task_id
        batch_size = self._args.batch_size
        io_dir = self._args.io_dir
        prm_data = self._args.prm
        tp = TaskPartitioner(io_dir, None, batch_size, tunnel_v=self._get_tunnel_v())

        mc_reference = np.load(prm_data)['V']
        V = np.load(tp.get_tqre_fn(task_id))['ReTouchQ']
        VM = uw.calculate_visibility_matrix2(V[0:batch_size], True,
                                             mc_reference[0:-1], False,
                                             # mc_reference[0:4], False,
                                             STEPPING_FOR_CONNECTIVITY)
        np.savez(tp.get_tqrevis_fn(task_id), VM=VM, Q=V, VMS=np.sum(VM, axis=-1))

    @staticmethod
    def _setup_parser_dump(subparsers):
        sp = subparsers.add_parser('dump', help='Dump different types of object.')
        ssp = sp.add_subparsers(dest='dump_object', help='Object type.')
        common = argparse.ArgumentParser(add_help=False)
        common.add_argument('vert_id', help='Vertex ID')
        common.add_argument('conf_sel', help='Configuration ID or Range')
        common.add_argument('in_dir', help='Input directory that stores the configurations')
        common.add_argument('out_dir', help='Output directory')

        dp = ssp.add_parser('samconf',
                help="dumps the sampled a range of predicted touch configurations.",
                parents=[common])
        dp2 = ssp.add_parser('tconf',
                help="dumps a range of colliding configurations",
                parents=[common]);
        dp2.add_argument('--signature', help='Only dump signature configurations', action='store_true')
        dp3 = ssp.add_parser('omplsam',
                help='dumps the samples to a text file or a npz file in OMPL convention.')
        dp3.add_argument('--withig', help='Add initial state and goal state as 0th/1th row in the output file', default=None)
        dp3.add_argument('in_dir', help='Input directory of predicted samples')
        dp3.add_argument('output_file', help='Output .txt file')

    def _dump_conf_common(self, fn_func, key_str):
        uw = self._uw
        vert_id = self._args.vert_id
        conf_sel = np.array(aux.range2list(self._args.conf_sel), dtype=np.int)
        print("Printing {}".format(conf_sel))
        io_dir = self._args.in_dir
        out_obj_dir = self._args.out_dir
        # Iterate through all ReTouchQ files
        batch_id = 0
        conf_base = 0
        largest = np.max(conf_sel)
        while conf_base <= largest:
            fn = fn_func(io_dir, vert_id=vert_id, batch_id=batch_id)
            if not os.path.exists(fn):
                print("Fatal: Cannot find file {}".format(fn))
                return
            Q = np.load(fn)[key_str]
            indices = np.where(np.logical_and(conf_sel >= conf_base, conf_sel < conf_base + len(Q)))
            indices = indices[0]
            # print("Indices {}".format(indices))
            Vall_list = None
            Fall_list = None
            Fall_base = 0
            for index in indices:
                conf_id = conf_sel[index]
                # print("conf_sel index {}".format(index))
                # print("conf_id {}".format(conf_id))
                rem = conf_id - conf_base
                # print("rem {}".format(rem))
                q = Q[rem]
                print("Getting geometry from file {}, offset {}".format(fn, rem))
                print("tq {}".format(q))
                V1,F1 = uw.get_robot_geometry(q, True)
                out_obj = "{}/rob-vert-{}-conf-{}.obj".format(out_obj_dir, vert_id, conf_id)
                pyosr.save_obj_1(V1, F1, out_obj)
                out_obj = "{}/env-vert-{}-conf-{}.obj".format(out_obj_dir, vert_id, conf_id)
                V2,F2 = uw.get_scene_geometry(q, True)
                pyosr.save_obj_1(V2, F2, out_obj)
                out_obj = "{}/union-vert-{}-conf-{}.obj".format(out_obj_dir, vert_id, conf_id)
                Va = np.concatenate((V1, V2), axis=0)
                F2 += V1.shape[0]
                Fa = np.concatenate((F1, F2), axis=0)
                if Vall_list is None:
                    # Include the environment
                    Vall_list = [V1,V2]
                    Fall_list = [F1,F2] # F2 has been rebased
                    Fall_base = V1.shape[0] + V2.shape[0]
                else:
                    # Exclude the environment, so env geo only included once
                    Vall_list.append(V1)
                    F1prime = F1 + Fall_base
                    Fall_list.append(F1prime)
                    Fall_base += V1.shape[0]
                #print("Dumping to {}".format(out_obj))
                pyosr.save_obj_1(Va, Fa, out_obj)
            Vall = np.concatenate(Vall_list, axis=0)
            Fall = np.concatenate(Fall_list, axis=0)
            out_obj = "{}/union-all.obj".format(out_obj_dir)
            pyosr.save_obj_1(Vall, Fall, out_obj)
            batch_id += 1
            conf_base += len(Q)

    def _dump_samconf(self):
        self._dump_conf_common(task_partitioner.touchq_fn, 'TOUCH_V')

    def _dump_tconf(self):
        if self._args.signature:
            self._dump_conf_common(task_partitioner.tqre_fn, 'SigReTouchQ')
        else:
            self._dump_conf_common(task_partitioner.tqre_fn, 'ReTouchQ')

    def _dump_omplsam(self):
        uw = self._uw
        in_dir = self._args.in_dir
        ofn = self._args.output_file
        ompl_q = None
        if self._args.withig is not None:
            config = configparser.ConfigParser()
            config.read([self._args.withig])
            def read_xyz(config, section, prefix):
                ret = np.zeros(shape=(3), dtype=np.float64)
                for i,suffix in enumerate(['x','y','z']):
                    ret[i] = config.getfloat(section, prefix + '.' + suffix)
                return ret
            def read_state(config, section, prefix):
                tr = read_xyz(config, section, prefix)
                rot_axis = read_xyz(config, section, prefix + '.axis')
                rot_angle = config.getfloat(section, prefix + '.theta')
                q = pyosr.compose_from_angleaxis(tr, rot_angle, rot_axis)
                return q.reshape((1, pyosr.STATE_DIMENSION))
            iq = read_state(config, 'problem', 'start')
            gq = read_state(config, 'problem', 'goal')
            ompl_q = np.concatenate((iq, gq), axis=0)
            assert pyosr.STATE_DIMENSION == 7, "FIXME: More flexible w-first to w-last"
            ompl_q[:, [6,3,4,5]] = ompl_q[:, [3,4,5,6]] # W-first (pyOSR) to W-last (OMPL)
            print("ig ompl_q {}".format(ompl_q.shape))
        if os.path.isdir(in_dir):
            tp = TaskPartitioner(in_dir, None, None, tunnel_v=self._get_tunnel_v())
            for task_id, _ in enumerate(iter(int, 1)):
                fn = tp.get_tqre_fn(task_id)
                print('probing {}'.format(fn))
                if not os.path.exists(fn):
                    break
                Q = np.load(fn)['ReTouchQ']
                QT = uw.translate_unit_to_ompl(Q)
                ompl_q = QT if ompl_q is None else np.concatenate((ompl_q, QT), axis=0)
        else:
            fn = in_dir
            if fn.endswith('.npz'):
                Q = np.load(fn)['ReTouchQ']
            elif fn.endswith('.txt'):
                Q = np.loadtxt(fn)
            else:
                raise NotImplemented("Unknown Extension of input file {}".format(fn))
            QT = uw.translate_unit_to_ompl(Q)
            ompl_q = QT if ompl_q is None else np.concatenate((ompl_q, QT), axis=0)
        nsample, scalar_per_sample = ompl_q.shape
        if ofn.endswith('.npz'):
            np.savez(ofn, OMPLV=ompl_q)
        else:
            with open(ofn, 'w') as f:
                f.write("{} {}\n".format(nsample, scalar_per_sample))
                for i in range(nsample):
                    for j in range(scalar_per_sample):
                        f.write('{:.17g} '.format(ompl_q[i,j]))
                    f.write('\n')

    def dump(self):
        getattr(self, '_dump_{}'.format(self._args.dump_object))()

    @staticmethod
    def _setup_parser_samstat(subparsers):
        sp = subparsers.add_parser('samstat',
                help="Parse the output from 'samvis' command, and show the statistics.")
        sp.add_argument('io_dir', help='Input/Output directory')
        sp.add_argument('prm', help='Sample set for visibility estimation')

    def samstat(self):
        uw = self._uw
        args = self._args
        class PerVertexStat(object):
            def __init__(self, vert_id):
                self.vms_array = None
                self.vert_id = vert_id

            def accumulate(self, dic):
                #print('Total V: {}'.format(np.sum(dic['VMS'])))
                if self.vms_array is None:
                    self.vms_array = dic['VMS']
                else:
                    self.vms_array = np.concatenate((self.vms_array, dic['VMS']), axis=0)

            def get_bins(self):
                return [float(i) * 0.005 for i in range(200)]

            def collect_percent(self, count):
                self.vms_pc = self.vms_array / float(count)
                # if self.vert_id == 0:
                    # print(self.vms_pc)
                nsample = float(len(self.vms_pc))
                bins = self.get_bins() #print(bins) hist, _ = np.histogram(self.vms_pc, bins)
                hist = hist/nsample * 100
                return hist

            def show(self, count):
                hist = self.collect_percent(count)
                hist_str = np.array_repr(hist, max_line_width=1024, precision=2, suppress_small=True)
                print("Histogram for vertex {}\n{}".format(self.vert_id, hist_str))

        tunnel_v = _get_tunnel_v()
        pvs_array = [ PerVertexStat(i) for i in range(len(tunnel_v)) ]
        io_dir = args.io_dir
        prm_data = args.prm
        mc_reference_count = np.load(prm_data)['V'].shape[0]
        # Probe batch size
        fn = task_partitioner.tqrevis_fn(io_dir, vert_id=0, batch_id=0)
        batch_size = int(np.load(fn)['Q'].shape[0])
        # With the probed batch size to create TaskPartitioner
        tp = TaskPartitioner(io_dir, None, batch_size, tunnel_v=_get_tunnel_v())
        task_id = 0
        while True:
            fn = tp.get_tqrevis_fn(task_id)
            if not os.path.exists(fn):
                break
            d = np.load(fn)
            batch_size = int(d['Q'].shape[0])
            vert_id = tp.get_vert_id(task_id)
            pvs_array[vert_id].accumulate(d)
            task_id += 1
        hists = []
        '''
        for pvs in pvs_array:
            pvs.show(mc_reference_count)
        '''
        for pvs in pvs_array:
            hists.append(pvs.collect_percent(mc_reference_count))
        hists_all = np.array(hists)
        hist_sum = np.sum(hists, axis=0)
        print(hist_sum)
        last_nz = np.nonzero(hist_sum)[0][-1] - 1 # [0]: access the tuple, [-1]: last element
        hists = hists_all[:,:last_nz]
        bins = np.array(pvs_array[0].get_bins()) * 100.0
        print(np.array_repr(bins[:last_nz], max_line_width=1024, precision=2, suppress_small=True))
        for hist in hists:
            hist_str = np.array_repr(hist, max_line_width=1024, precision=2, suppress_small=True)
            print(hist_str)
        with open('{}/hist.csv'.format(io_dir), 'w') as f:
            f.write(',')
            for s,e in zip(bins[:-1], bins[1:]):
                f.write('{:.3f}%-{:.3f}%'.format(s, e))
                f.write(',')
            f.write('\n')
            for i,hist in enumerate(hists):
                f.write('{},'.format(i))
                for h in hist:
                    f.write('{:4.2f}%'.format(h))
                    f.write(',')
                f.write('\n')
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        X1 = bins[:last_nz]
        X2 = np.array([i for i in range(last_nz)])
        Y1 = np.array([i for i in range(len(hists)-1)])
        Y2 = np.array([i for i in range(len(hists)-1)])
        X1, Y1 = np.meshgrid(X1, Y1)
        X2, Y2 = np.meshgrid(X2, Y2)
        print(hists_all.shape)
        Z = hists_all[Y2, X2]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        from matplotlib import cm
        surf = ax.plot_surface(Y1, X1, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    @staticmethod
    def _setup_parser_util(subparsers):
        sp = subparsers.add_parser('util', help='Utility functions within this infrastructure.')
        ssp = sp.add_subparsers(dest='util_name', help='Name of utility function')
        unit2ompl = ssp.add_parser('unit2ompl', help='Read unit states from test file and convert them to ompl states')
        unit2ompl.add_argument('textin', help='Input text file')
        unit2ompl.add_argument('textout', help='Output text file')
        unit2ompl.add_argument('--angleaxis', help='Use angle axis as the rotation component (the default is quaternion)', action='store_true')
        unit2ompl = ssp.add_parser('ompl2unit', help='Read unit states from test file and convert them to ompl states')
        unit2ompl.add_argument('textin', help='Input text file')
        unit2ompl.add_argument('textout', help='Output text file')

    def _util_unit2ompl(self):
        uw = self._uw
        ifn = self._args.textin
        ofn = self._args.textout
        Q = np.loadtxt(ifn)
        ompl_q = uw.translate_unit_to_ompl(Q, to_angle_axis=self._args.angleaxis)
        if ofn.endswith('.npz'):
            np.savez(ofn, OMPLV=ompl_q)
        else:
            np.savetxt(ofn, ompl_q, fmt='%.17g')

    def _util_ompl2unit(self):
        uw = self._uw
        ifn = self._args.textin
        ofn = self._args.textout
        Q = np.loadtxt(ifn)
        unit_q = uw.translate_ompl_to_unit(Q)
        if ofn.endswith('.npz'):
            np.savez(ofn, UNITV=ompl_q)
        else:
            np.savetxt(ofn, unit_q, fmt='%.17g')

    def util(self):
        getattr(self, '_util_{}'.format(self._args.util_name))()

    @staticmethod
    def _setup_parser_screen(subparsers):
        sp = subparsers.add_parser('screen', help='Screening the samples to cluster nearby samples')
        sp.add_argument('--method', help='Choose screening method', type=int, choices=[0], required=True)
        sp.add_argument('prescreen', help='Input file in NPZ')
        sp.add_argument('connectivity', help='Connectivity file precalculated by condor-visibility-matrix2.py and asvm.py, in .mat format')
        sp.add_argument('postscreen', help='Output file in NPZ')

    def screen(self):
        getattr(self, '_screen_{}'.format(self._args.method))()

    def _screen_0(self):
        d = np.load(self._args.prescreen)
        samples = d[d.keys()[0]]
        cd = loadmat(self._args.connectivity)
        cm = cd['VM']
        outfn = self._args.postscreen
        vert_ids = [i for i in range(samples.shape[0])]
        djs = disjoint_set.DisjointSet(vert_ids)
        nz_rows, nz_cols = cm.nonzero()
        for r,c in zip(nz_rows, nz_cols):
            djs.union(r,c)
        screened = samples[djs.get_roots()]
        print('screened array shape {}'.format(screened.shape))
        np.savez(outfn, ReTouchQ=screened)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--puzzle', help='choose puzzle to solve', required=True)
    subparsers = parser.add_subparsers(dest='command')
    for fn in  ['run', 'isect', 'uvproj', 'uvrender', 'uvmerge', 'atlas2prim', 'useatlas', 'sample', 'sample_enumaxis', 'samvis', 'dump', 'samstat', 'util', 'screen']:
        getattr(TouchConfiguration, '_setup_parser_'+fn)(subparsers)

    args = parser.parse_args()
    touch_conf = TouchConfiguration(args)
    getattr(touch_conf, args.command)()

    return

if __name__ == '__main__':
    main()
