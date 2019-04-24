#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import subprocess
import pathlib
import numpy as np
from scipy.misc import imsave
import h5py

from . import util
from . import choice_formatter
from . import matio
from . import partt
from . import touchq_util
from . import texture_format
import pyosr

_TOUCH_SCRATCH = join(util.CONDOR_SCRATCH, 'training_key_touch')
_ISECT_SCRATCH = join(util.CONDOR_SCRATCH, 'training_key_isect')
_UVPROJ_SCRATCH = util.UV_DIR

def sample_touch(args, ws):
    scratch_dir = ws.condor_ws(_TOUCH_SCRATCH)
    if args.only_wait:
        condor.local_wait(scratch_dir)
        return
    keys = matio.load(ws.condor_ws(util.KEY_FILE))['KEYQ']
    tqn = ws.config.getint('TrainingWeightChart', 'TouchSample')
    nkey = keys.shape[0]
    task_shape = (nkey, tqn) # 2D, we may specify large number of samples per key configuration
    total_chunks = partt.guess_chunk_number(task_shape,
            ws.config.getint('DEFAULT', 'CondorQuota') * 2,
            ws.config.getint('TrainingWeightChart', 'TouchSampleGranularity'))
    if total_chunks > 1 and args.task_id is None:
        # Submit a Condor job
        args = ['facade.py',
                'preprocess_surface',
                '--stage',
                'sample_touch'
                '--task_id',
                '$(Process)']
        condor.local_submit(ws,
                            util.PYTHON,
                            iodir=scratch_dir,
                            arguments=args,
                            instances=total_chunks,
                            wait=True)
    else:
        uw = ws.condor_unit_world(util.TRAINING_DIR)
        # Do the actual work (either no need to partition, or run from HTCondor)
        task_id = 0 if total_chunks == 1 else args.task_id
        tindices = partt.get_task_chunk(task_shape, total_chunks, task_id)
        # ki: Key index. si: Sample index
        from_key_index = []
        from_keys = []
        free_qs = []
        touch_qs = []
        is_inf = []
        for ki, _ in tindices:
            key = keys[ki]
            tup = touchq_util.sample_one_touch(uw, key, uw.recommended_cres)
            from_key_index.append(ki)
            from_keys.append(key)
            free_qs.append(tup[0])
            touch_qs.append(tup[1])
            is_inf.append(tup[2])
        # Note we need to pad zeros because we want to keep the order
        task_id_str = util.padded(task_id, total_chunks)
        tq_out = ws.condor_ws(_TOUCH_SCRATCH, 'touchq_batch-{}.npz'.format(task_id_str))
        np.savez(tq_out, FROM_VI=from_key_index, FROM_V=from_keys, FREE_V=free_qs, TOUCH_V=touch_qs, IS_INF=is_inf)


def isect_geometry(args, ws):
    '''
    Mostly copied from sample_touch()
    It might be over enginerring to do de-duplication here
    '''
    prev_scratch_dir = ws.condor_ws(_TOUCH_SCRATCH)
    scratch_dir = ws.condor_ws(_ISECT_SCRATCH)
    if args.only_wait:
        condor.local_wait(scratch_dir)
        return
    # We need a consistent order for the task partitioning.
    fn_list = sorted(pathlib.Path(prev_scratch_dir).glob('touchq_batch-*.npz'))
    tqn = ws.config.getint('TrainingWeightChart', 'TouchSample')
    task_shape = (len(fn_list), tqn) # 2D, ignore some "holes" (connecting to open space)
    total_chunks = partt.guess_chunk_number(task_shape,
            ws.config.getint('DEFAULT', 'CondorQuota') * 4,
            ws.config.getint('TrainingWeightChart', 'MeshBoolGranularity'))
    if args.task_id is None:
        np.savez(join(scratch_dir, "taskinfo.npz"),
                 TASK_SHAPE=list(task_shape),
                 TOTAL_CHUNKS=total_chunks)
    if total_chunks > 1 and args.task_id is None:
        # Submit a Condor job
        args = ['facade.py',
                'preprocess_surface',
                '--stage',
                'isect_geometry'
                '--task_id',
                '$(Process)']
        condor.local_submit(ws,
                            util.PYTHON,
                            iodir=scratch_dir,
                            arguments=args,
                            instances=total_chunks,
                            wait=True)
    else:
        uw = ws.condor_unit_world(util.TRAINING_DIR)
        # Do the actual work (either no need to partition, or run from HTCondor)
        task_id = 0 if total_chunks == 1 else args.task_id
        tindices = partt.get_task_chunk(task_shape, total_chunks, task_id)
        cache_file_index = None
        cache_file = None
        cache_from = None
        cache_tqs = None
        cache_inf = None
        f = h5py.File(join(scratch_dir, 'isect_batch-{}.hdf5'.format(task_id)), 'a')
        for index, (fni, si) in enumerate(tindices):
            if cache_file_index != fni:
                cache_file = matio.load(fn_list[fni])
                cache_fromi = cache_file['FROM_VI']
                cache_from = cache_file['FROM_V']
                cache_tqs = cache_file['TOUCH_V']
                cache_inf = cache_file['IS_INF']
                cache_file_index = fni
            if cache_inf[si]:
                continue
            tq = cache_tqs[si]
            V, F = uw.intersecting_geometry(tq, True)
            index_id_str = util.padded(index, len(tindices))
            hdf5_overwrite(f, '{}/V'.format(index_id_str), V)
            hdf5_overwrite(f, '{}/F'.format(index_id_str), F)
            hdf5_overwrite(f, '{}/tq'.format(index_id_str), tq)
            hdf5_overwrite(f, '{}/from'.format(index_id_str), cache_from[si])
            hdf5_overwrite(f, '{}/fromi'.format(index_id_str), cache_fromi[si])
        f.close()


def uvproject(args, ws):
    prev_scratch_dir = ws.condor_ws(_ISECT_SCRATCH)
    scratch_dir = ws.condor_ws(_UVPROJ_SCRATCH)
    if args.only_wait:
        condor.local_wait(scratch_dir)
        return
    prev_taskinfo = np.load(join(prev_scratch_dir, 'taskinfo.npz'))
    prev_shape = prev_taskinfo['TASK_SHAPE']
    prev_chunks = prev_taskinfo['TOTAL_CHUNKS']
    per_file_geometry = len(partt.get_task_chunk(prev_shape, prev_chunks, 0))
    fn_list = sorted(pathlib.Path(prev_scratch_dir).glob('isect_batch-*.hdf5'))
    task_shape = (len(fn_list), per_file_geometry)
    total_chunks = partt.guess_chunk_number(task_shape,
            ws.config.getint('DEFAULT', 'CondorQuota') * 2,
            ws.config.getint('TrainingWeightChart', 'UVProjectGranularity'))
    if total_chunks > 1 and args.task_id is None:
        # Submit a Condor job
        args = ['facade.py',
                'preprocess_surface',
                '--stage',
                'uvproject'
                '--task_id',
                '$(Process)']
        condor.local_submit(ws,
                            util.PYTHON,
                            iodir=scratch_dir,
                            arguments=args,
                            instances=total_chunks,
                            wait=True)
    else:
        uw = ws.condor_unit_world(util.TRAINING_DIR)
        tindices = partt.get_task_chunk(task_shape, total_chunks, task_id)
        f = h5py.File(join(scratch_dir, 'uv_batch-{}.hdf5'.format(task_id)), 'r')
        cache_file_index = None
        for index, (fni, si) in enumerate(tindices):
            si_str = util.padded(si, per_file_geometry)
            if cache_file_index != fni:
                cache_file = matio.load(fn_list[fni])
                cache_file_index = fni
            gpn = '{}/'.format(si_str)
            if gpn not in cache_file:
                continue
            grp = cache_file[gpn]
            tq = grp['tq']
            V = grp['V']
            F = grp['F']
            IF, IBV = uw.intersecting_to_robot_surface(tq, True, V, F)
            hdf5_overwrite(f, gpn+'V.rob', IBV)
            hdf5_overwrite(f, gpn+'F.rob', IF)
            IF, IBV = uw.intersecting_to_model_surface(tq, True, V, F)
            hdf5_overwrite(f, gpn+'V.env', IBV)
            hdf5_overwrite(f, gpn+'F.env', IF)
            hdf5_overwrite(f, gpn+'tq', tq)
            hdf5_overwrite(f, gpn+'fromi', grp[fromi])
        f.close()


def uvrender(args, ws):
    uvproj_dir = ws.local_ws(_UVPROJ_SCRATCH)
    keys = matio.load(ws.condor_ws(util.KEY_FILE))['KEYQ']
    uvproj_list = sorted(pathlib.Path(uvproj_dir).glob('uv_batch-*.hdf5'))
    r = util.create_offscreen_renderer(ws.local_ws(util.TRAINING_DIR, PUZZLE_CFG_FILE))
    TYPE_TO_FLAG = {'rob' : r.BARY_RENDERING_ROBOT,
                    'env' : r.BARY_RENDERING_SCENE }
    for rname, rflag in TYPE_TO_FLAG.items():
        chart_resolution = np.array([ws.chart_resolution, ws.chart_resolution], dtype=np.int32)
        afb = None
        afb_uw = None # Uniform weight
        for fn in uvproj_list:
            f = h5py.File(fn, 'r')
            for grn, grp in f.items():
                IBV = grp['V.{}'.format(rname)][:]
                IF = grp['F.{}'.format(rname)][:]
                tq = grp['tq']
                iq = keys[grp['fromi']]
                r.clear_barycentric(rflag)
                r.add_barycentric(IF, IBV, rflag)
                fb = r.render_barycentric(rflag, chart_resolution, svg_fn='')
                uniform_weight = texture_format.texture_to_file(fb.astype(np.float32))
                distance = np.clip(pyosr.distance(tq, iq), 1e-4, None)
                w = uniform_weight * (1.0 / distance)
                if afb is None:
                    afb = w
                    afb_uw = uniform_weight
                else:
                    afb += w
                    afb_uw += uniform_weight
                    np.clip(afb_uw, 0, 1.0, out=afb_uw) # afb_uw is supposed to be binary
        np.savez(ws.local_ws(util.TRAINING_DIR, '{}_chart.npz'.format(rname)),
                 WEIGHTED=afb,
                 UNIFORM_WEIGHTED=abf_uw)
        rgb = np.zeros(list(afb.shape) + [3])
        rgb[...,1] = afb
        imsave(ws.local_ws(util.TRAINING_DIR, '{}_chart.png'.format(rname)), rgb)
        rgb[...,1] = afb_uw
        imsave(ws.local_ws(util.TRAINING_DIR, '{}_chart_uniform_weight.png'.format(rname)), rgb)


def screen_weight(args, ws):
    for geo_type in ['rob', 'env']:
        d = matio.load(ws.local_ws(util.TRAINING_DIR, '{}_chart.npz'.format(geo_type)))
        img = d['WEIGHTED'] # Unlike condor_touch_configuration, we only have one file here
        LOOPS = 2 # Emperical number
        for i in range(LOOPS):
            nzi = np.nonzero(img)
            print("[screen_weight] geo {} loop {}: nz count {}".format(geo_type, i, len(nzi[0])))
            nz = img[nzi]
            m = np.mean(nz)
            img[img < m] = 0.0
            print("[screen_weight] geo {} loop {}: sum {}".format(geo_type, i+1, np.sum(img)))
        imsave(ws.local_ws(util.TRAINING_DIR, '{}_chart_screened.png'.format(geo_type)), img)


function_dict = {
        'sample_touch' : sample_touch,
        'isect_geometry' : isect_geometry,
        'uvproject' : uvproject,
        'uvrender' : uvrender,
        'screen_weight' : screen_weight,
}

def setup_parser(subparsers):
    p = subparsers.add_parser('preprocess_surface',
                              help='Preprocessing step, to generate training data',
                              formatter_class=choice_formatter.Formatter)
    p.add_argument('--stage',
                   choices=list(function_dict.keys()),
                   help='R|Possible stages:\n'+'\n'.join(list(function_dict.keys())),
                   default='',
                   metavar='')
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--task_id', help='Feed $(Process) from HTCondor', type=int, default=None)


def run(args):
    if args.stage in function_dict:
        ws = util.Workspace(args.dir)
        function_dict[args.stage](args, ws)
    else:
        print("Unknown preprocessing pipeline stage {}".format(args.stage))

def _remote_command(ws, cmd, auto_retry=True):
    ws.remote_command(ws.condor_host,
                      ws.condor_exec(),
                      ws.condor_ws(),
                      'preprocess_surface', cmd, auto_retry=auto_retry)

def remote_sample_touch(ws):
    _remote_command(ws, 'sample_touch')

def remote_isect_geometry(ws):
    _remote_command(ws, 'isect_geometry')

def remote_uvproject(ws):
    _remote_command(ws, 'uvproject')

def autorun(args):
    ws = util.Workspace(args.dir)
    pdesc = collect_stages()
    for _,func in pdesc:
        func(ws)

def collect_stages():
    return [ ('sample_touch', remote_sample_touch),
             ('isect_geometry', remote_isect_geometry),
             ('uvproject', remote_uvproject),
             ('fetch_groundtruth', lambda ws: ws.fetch_condor(util.UV_DIR + '/')),
             ('uvrender', lambda ws: uvrender(None, ws)),
             ('screen_weight', lambda ws: screen_weight(None, ws))
           ]
