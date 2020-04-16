#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isdir, isfile
import copy
import pathlib
import numpy as np
import argparse
from imageio import imwrite as imsave

from . import util
from . import matio
from . import parse_ompl
from . import atlas
from . import partt
from .file_locations import FEAT_PRED_SCHEMES, RAW_KEY_PRED_SCHEMES, KEY_PRED_SCHEMES, SCHEME_TO_FMT, SCHEME_FEAT_NPZ_KEY, FileLocations

class TmpDriverArgs(object):
    pass

def read_roots(args):
    uw = util.create_unit_world(args.puzzle_fn)
    ompl_q = matio.load(args.roots, key=args.roots_key)
    print("OMPL_Q {}".format(ompl_q.shape))
    if args.to_vanilla:
        store_q = uw.translate_ompl_to_vanilla(ompl_q)
    else:
        store_q = uw.translate_ompl_to_unit(ompl_q)
    matio.savetxt(args.out, store_q)

def write_roots(args):
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    unit_q = matio.load(args.roots, key=args.input_key)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        uw = util.create_unit_world(puzzle_fn)
        d = {}
        ompl_q = uw.translate_unit_to_ompl(unit_q)
        if not args.noig:
            cfg, config = parse_ompl.parse_simple(puzzle_fn)
            iq = parse_ompl.tup_to_ompl(cfg.iq_tup)
            gq = parse_ompl.tup_to_ompl(cfg.gq_tup)
            ompl_q = np.concatenate((iq, gq, ompl_q), axis=0)
        d[args.roots_key] = ompl_q
        key_fn = ws.screened_keyconf_prediction_file(puzzle_name)
        np.savez(key_fn, **d)
        util.log('[tools][write_roots] write {} roots to {}'.format(ompl_q.shape, key_fn))

def _visgt(args):
    ws = util.Workspace(args.dir)
    cfg, config = parse_ompl.parse_simple(ws.training_puzzle)
    puzzle_fn = None
    puzzle_fn = cfg.env_fn if args.geo_type == 'env' else puzzle_fn
    puzzle_fn = cfg.rob_fn if args.geo_type == 'rob' else puzzle_fn
    p = pathlib.Path(puzzle_fn)
    util.shell(['./vistexture', puzzle_fn, str(p.with_suffix('.png')),])

def visenvgt(args):
    args.geo_type = 'env'
    _visgt(args)

def visrobgt(args):
    args.geo_type = 'rob'
    _visgt(args)

def visnnpred(args):
    ws = util.Workspace(args.dir)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        if args.puzzle_name and puzzle_name != args.puzzle_name:
            continue
        cfg, config = parse_ompl.parse_simple(puzzle_fn)
        """
        env_npz = ws.atex_prediction_file(puzzle_fn, 'env', trial_override=args.current_trial)
        util.shell(['./atex2green.py', env_npz])
        rob_npz = ws.atex_prediction_file(puzzle_fn, 'rob', trial_override=args.current_trial)
        util.shell(['./atex2green.py', rob_npz])
        p = pathlib.Path(env_npz)
        util.shell(['./vistexture', cfg.env_fn, str(p.with_suffix('.png'))])
        p = pathlib.Path(rob_npz)
        util.shell(['./vistexture', cfg.rob_fn, str(p.with_suffix('.png'))])
        """
        env_npz = ws.atex_prediction_file(puzzle_fn, 'env', trial_override=args.current_trial)
        rob_npz = ws.atex_prediction_file(puzzle_fn, 'rob', trial_override=args.current_trial)
        p = pathlib.Path(env_npz)
        util.shell(['./vistexture.py', cfg.env_fn, str(p)])
        p = pathlib.Path(rob_npz)
        util.shell(['./vistexture.py', cfg.rob_fn, str(p)])

def visnnsample(args):
    import pyosr
    ws = util.Workspace(args.dir)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        cfg, config = parse_ompl.parse_simple(puzzle_fn)
        if args.puzzle_name and puzzle_name != args.puzzle_name:
            continue
        p = pathlib.Path(puzzle_fn)
        d = p.parents[0]
        env_tex_fn = str(d / 'env-atex-dbg.png')
        rob_tex_fn = str(d / 'rob-atex-dbg.png')
        if args.update:
            rob_sampler = atlas.AtlasSampler(ws.local_ws(util.TESTING_DIR, puzzle_name, 'rob-a2p.npz'),
                                             ws.local_ws(util.TESTING_DIR, puzzle_name, 'rob-atex.npz'),
                                             'rob', pyosr.UnitWorld.GEO_ROB)
            env_sampler = atlas.AtlasSampler(ws.local_ws(util.TESTING_DIR, puzzle_name, 'env-a2p.npz'),
                                             ws.local_ws(util.TESTING_DIR, puzzle_name, 'env-atex.npz'),
                                             'env', pyosr.UnitWorld.GEO_ENV)
            env_sampler.debug_surface_sampler(env_tex_fn)
            rob_sampler.debug_surface_sampler(rob_tex_fn)
        util.shell(['./vistexture', cfg.env_fn, env_tex_fn])
        util.shell(['./vistexture', cfg.rob_fn, rob_tex_fn])

def vistraj(args):
    ws = util.Workspace(args.dir)
    ws.fetch_condor(util.TRAJECTORY_DIR + '/')
    trajfn = ws.local_ws(util.TRAJECTORY_DIR, 'traj_{}.npz'.format(args.traj_id))
    utrajfn = ws.local_ws(util.TRAJECTORY_DIR, 'traj_{}.unit.txt'.format(args.traj_id))
    uw = util.create_unit_world(ws.training_puzzle)
    ompl_q = matio.load(trajfn, key='OMPL_TRAJECTORY')
    unit_q = uw.translate_ompl_to_unit(ompl_q)
    matio.savetxt(utrajfn, unit_q)
    cfg, config = parse_ompl.parse_simple(ws.training_puzzle)
    util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, utrajfn, '0.5'])

def plotstat(args):
    import matplotlib
    import matplotlib.pyplot as plt
    '''
    Load keys
    '''
    ws = util.Workspace(args.dir)
    if not args.noupdate:
        ws.fetch_condor(util.TRAINING_DIR, 'KeyCan.npz')
    kq = matio.load(ws.local_ws(util.KEY_FILE), key='KEYQ')
    uw = util.create_unit_world(ws.training_puzzle)
    '''
    Load stats
    '''
    d = matio.load(ws.local_ws(util.KEY_FILE))
    all_st = d['_STAT_VALUES']
    st_keymap = d['_STAT_KEYS']
    top_k_ids = d['_TOP_K_PER_TRAJ']
    st = all_st[args.offset]
    util.log('st shape {} from Traj {} top_k_ids {}'.format(st.shape, st_keymap[args.offset], top_k_ids[args.offset]))
    medians, maxs, mins, means, stddevs = np.transpose(st[:args.trim,:])
    indices=range(medians.shape[0])

    top_k = ws.config.getint('TrainingKeyConf', 'KeyConf')
    util.log("Saving keyq[{} : {}]".format(top_k * args.offset, top_k * (args.offset + 1)))
    matio.savetxt('keyq.unit.txt', kq[top_k * args.offset : top_k * (args.offset + 1)])
    # matio.savetxt('keyq.unit.txt', kq[:10])
    '''
    Vis stats
    '''
    plt.plot(indices, means, 'r-')
    plt.plot(indices, medians, 'b-')
    plt.plot(indices, maxs, 'g-')
    plt.show()

    cfg, config = parse_ompl.parse_simple(ws.training_puzzle)
    util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'keyq.unit.txt', '0.5'])

def visclearance(args):
    ws = util.Workspace(args.dir)
    '''
    FIXME: this code is copied from preprocess_key.estimate_clearance_volume.
    '''
    candidate_file = ws.local_ws(util.MT_KEY_CANDIDATE_FILE)
    cf = matio.load(candidate_file)
    trajs = sorted(list(cf.keys()))
    ntraj = len(trajs)
    nq = cf[trajs[0]].shape[0]
    task_shape = (ntraj, nq)
    total_chunks = partt.guess_chunk_number(task_shape,
            ws.config.getint('SYSTEM', 'CondorQuota') * 2,
            ws.config.getint('TrainingKeyConf', 'ClearanceTaskGranularity'))
    task_partition = partt.get_task_partition(task_shape, total_chunks)
    tgt = (args.traj_id, args.point_id)
    for task_id, task_content in enumerate(task_partition):
        if tgt not in task_content:
            continue
        task_id_str = util.padded(task_id, total_chunks)
        fn = ws.local_ws(util.PREP_KEY_CAN_SCRATCH,
                         'unitary_clearance_from_keycan-batch_{}.hdf5.xz'.format(task_id_str))
        util.log('[visclearance] find {} at {}'.format(tgt, fn))
        d = matio.load(fn)
        traj_name = trajs[args.traj_id]
        qi_str = util.padded(args.point_id, nq)
        gpn = traj_name + '/' + qi_str + '/'
        from_v = np.reshape(d[gpn+'FROM_V'], (1,7))
        free_vertices = d[gpn+'FREE_V']
        # matio.savetxt('keyq.unit.txt', from_v)
        matio.savetxt('keyq.unit.txt', free_vertices)
        cfg, config = parse_ompl.parse_simple(ws.training_puzzle)
        util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'keyq.unit.txt', '0.5'])
        break

def viskey(args):
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    def puzgen(args):
        if args.puzzle_name and args.puzzle_name == 'train':
            yield ws.training_puzzle, 'train', ws.local_ws(util.KEY_FILE)
        else:
            for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
                if args.scheme is not None:
                    """
                    Autorun6 nameing protocol: with --scheme
                    """
                    fl = FileLocations(args, ws, puzzle_name, scheme=args.scheme)
                    if args.unscreened:
                        key_fn = fl.raw_key_fn
                    else:
                        key_fn = fl.screened_key_fn
                else:
                    """
                    Autorun4 nameing protocol: without --scheme
                    """
                    if args.unscreened:
                        key_fn = ws.keyconf_prediction_file(puzzle_name)
                    else:
                        key_fn = ws.screened_keyconf_prediction_file(puzzle_name)
                if not isfile(key_fn):
                    util.log("[viskey] Could not find {}".format(key_fn))
                    continue
                util.log(f'[viskey] loading {key_fn}')
                yield puzzle_fn, puzzle_name, key_fn
        return
    for puzzle_fn, puzzle_name, key_fn in puzgen(args):
        d = matio.load(key_fn)
        keys = d['KEYQ_OMPL']
        uw = util.create_unit_world(puzzle_fn)
        if args.range:
            keys = keys[util.rangestring_to_list(args.range)]
        ukeys = uw.translate_ompl_to_unit(keys)
        matio.savetxt('viskey.tmp.txt', ukeys)

        cfg, _ = parse_ompl.parse_simple(puzzle_fn)
        cmd = ['./vispath', cfg.env_fn, cfg.rob_fn, 'viskey.tmp.txt', '0.5']
        type2flag = {'rob':uw.GEO_ROB, 'env':uw.GEO_ENV}
        """
        if 'ENV_KEYID' in d:
            env_keyid = d['ENV_KEYID'].reshape(-1)
            rob_keyid = d['ROB_KEYID'].reshape(-1)
            def _load_kps(geo_type):
                geo_flag = type2flag[geo_type]
                kps_fn = ws.keypoint_prediction_file(puzzle_name, geo_type)
                d = matio.load(kps_fn)
                pts = util.access_keypoints(d)
                unit_imp_1 = uw.translate_vanilla_pts_to_unit(geo_flag, pts[:, 0:3])
                unit_imp_2 = uw.translate_vanilla_pts_to_unit(geo_flag, pts[:, 3:6])
                return np.concatenate((unit_imp_1, unit_imp_2), axis=1)
            if True:
                env_kps = _load_kps('env')[env_keyid, :]
                rob_kps = _load_kps('rob')[rob_keyid, :]
                karma = np.concatenate((env_kps, rob_kps), axis=1)
            else:
                env_kps = _load_kps('env')
                rob_kps = _load_kps('rob')
                if True:
                    cap = min(env_kps.shape[0], rob_kps.shape[0])
                    env_kps = env_kps[:cap]
                    rob_kps = rob_kps[:cap]
                    karma = np.concatenate((env_kps, rob_kps), axis=1)
                else:
                    # karma = np.concatenate((env_kps, env_kps), axis=1)
                    karma = np.concatenate((rob_kps, rob_kps), axis=1)
            matio.savetxt('viskey.karma.txt', karma)
            matio.savetxt('viskey.env_kps.txt', env_kps[:,:])
            matio.savetxt('viskey.rob_kps.txt', rob_kps[:,:])
            cmd.append('viskey.karma.txt')
        """
        util.shell(cmd)

def visfeat(args):
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        cfg, _ = parse_ompl.parse_simple(puzzle_fn)
        uw = util.create_unit_world(puzzle_fn)
        if args.scheme is not None: # New code path, enabled by --scheme
            fl = FileLocations(args, ws, puzzle_name, scheme=args.scheme)
            for geo_type, geo_flag in zip(['rob', 'env'], [uw.GEO_ROB, uw.GEO_ENV]):
                pts = matio.load(fl.get_feat_pts_fn(geo_type=geo_type), key=fl.feat_npz_key)
                unit_imp_1 = uw.translate_vanilla_pts_to_unit(geo_flag, pts[:, 0:3])
                unit_imp_2 = uw.translate_vanilla_pts_to_unit(geo_flag, pts[:, 3:6])
                matio.savetxt('visfeat.tmp.txt', np.concatenate((unit_imp_1, unit_imp_2), axis=0))
                util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'visfeat.tmp.txt', '0.5'])
            continue
        for geo_type, geo_flag in zip(['rob', 'env'], [uw.GEO_ROB, uw.GEO_ENV]):
            kps_fn = ws.keypoint_prediction_file(puzzle_name, geo_type)
            d = matio.load(kps_fn)
            for key_name in ['KEY_POINT_AMBIENT', 'NOTCH_POINT_AMBIENT']:
                if key_name not in d:
                    continue
                pts = d[key_name]
                if pts.shape[0] == 0:
                    continue
                unit_imp_1 = uw.translate_vanilla_pts_to_unit(geo_flag, pts[:, 0:3])
                unit_imp_2 = uw.translate_vanilla_pts_to_unit(geo_flag, pts[:, 3:6])
                matio.savetxt('visfeat.tmp.txt', np.concatenate((unit_imp_1, unit_imp_2), axis=0))
                util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'visfeat.tmp.txt', '0.5'])

def _old_visfeat(args):
    import pygeokey
    class WorkerArgs(object):
        pass
    ws = util.Workspace(args.dir)
    #for puzzle_fn, puzzle_name in ws.training_puzzle_generator():
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        cfg, config = parse_ompl.parse_simple(puzzle_fn)
        wag = WorkerArgs()
        wag.dir = ws.dir
        wag.current_trial = ws.current_trial
        wag.puzzle_fn = puzzle_fn
        wag.puzzle_name = puzzle_name
        wag.env_fn = cfg.env_fn
        wag.rob_fn = cfg.rob_fn
        wag.geo_type = 'rob'
        wag.geo_fn = cfg.rob_fn
        kpp = pygeokey.KeyPointProber(wag.geo_fn)
        while True:
            pts = kpp.probe_key_points(args.pairs)
            if pts.shape[0] > 0:
                break
            util.log("Found 0 key points, try again")
        uw = util.create_unit_world(wag.puzzle_fn)
        unit_imp_1 = uw.translate_vanilla_pts_to_unit(uw.GEO_ROB, pts[:, 0:3])
        unit_imp_2 = uw.translate_vanilla_pts_to_unit(uw.GEO_ROB, pts[:, 3:6])
        matio.savetxt('visfeat.tmp.txt', np.concatenate((unit_imp_1, unit_imp_2), axis=0))
        matio.savetxt('visfeat.vanilla.txt', np.concatenate((pts[:, 0:3], pts[:,3:6]), axis=0))
        util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'visfeat.tmp.txt', '0.5'])


def visnotch(args):
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        cfg, config = parse_ompl.parse_simple(puzzle_fn)
        fl = FileLocations(args, ws, puzzle_name)
        fl.update_scheme('nt')
        uw = util.create_unit_world(puzzle_fn)
        for geo_type, geo_flag in zip(['rob', 'env'], [uw.GEO_ROB, uw.GEO_ENV]):
            key_fn = fl.get_feat_pts_fn(geo_type)
            pts = matio.load(key_fn)[fl.feat_npz_key]
            unit_imp_1 = uw.translate_vanilla_pts_to_unit(geo_flag, pts[:, 0:3])
            unit_imp_2 = uw.translate_vanilla_pts_to_unit(geo_flag, pts[:, 3:6])
            matio.savetxt('visfeat.tmp.txt', np.concatenate((unit_imp_1, unit_imp_2), axis=0))
            util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'visfeat.tmp.txt', '0.5'])
    """
    # FIXME: this function is mostly copied from visfeat
    import pygeokey
    class WorkerArgs(object):
        pass
    ws = util.Workspace(args.dir)
    #for puzzle_fn, puzzle_name in ws.training_puzzle_generator():
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        cfg, config = parse_ompl.parse_simple(puzzle_fn)
        wag = WorkerArgs()
        wag.dir = ws.dir
        wag.current_trial = ws.current_trial
        wag.puzzle_fn = puzzle_fn
        wag.puzzle_name = puzzle_name
        wag.env_fn = cfg.env_fn
        wag.rob_fn = cfg.rob_fn
        wag.geo_type = 'env'
        wag.geo_fn = cfg.env_fn
        kpp = pygeokey.KeyPointProber(wag.geo_fn)
        while True:
            pts = kpp.probe_notch_points()
            if pts.shape[0] > 0:
                break
            util.log("Found 0 key points, try again")
        uw = util.create_unit_world(wag.puzzle_fn)
        util.log("Found {} key points".format(pts.shape[0]))
        unit_imp_1 = uw.translate_vanilla_pts_to_unit(uw.GEO_ROB, pts[:, 0:3])
        unit_imp_2 = uw.translate_vanilla_pts_to_unit(uw.GEO_ROB, pts[:, 3:6])
        matio.savetxt('visfeat.tmp.txt', np.concatenate((unit_imp_1, unit_imp_2), axis=0))
        matio.savetxt('visfeat.vanilla.txt', np.concatenate((pts[:, 0:3], pts[:,3:6]), axis=0))
        util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'visfeat.tmp.txt', '0.5'])
    """

def vistouchv(args):
    ws = util.Workspace(args.dir)
    candidate_file = ws.local_ws(util.KEY_CANDIDATE_FILE)
    Qs = matio.load(candidate_file)['OMPL_CANDIDATES']
    nq = Qs.shape[0]
    qi_str = util.padded(args.key_id, nq)
    clearance_fn = ws.local_ws(util.PREP_KEY_CAN_SCRATCH, 'unitary_clearance_from_keycan-{}.npz'.format(qi_str))
    ukeys = matio.load(clearance_fn)['TOUCH_V']
    matio.savetxt('vistouchv.tmp.txt', ukeys)
    cfg, _ = parse_ompl.parse_simple(ws.training_puzzle)
    util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'vistouchv.tmp.txt', '0.5'])

def animate(args):
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        cfg, config = parse_ompl.parse_simple(puzzle_fn)
        if args.puzzle_name and puzzle_name != args.puzzle_name:
            continue
        sol_out = ws.solution_file(puzzle_name, type_name='unit')
        if os.path.exists(sol_out):
            if args.range:
                li = util.rangestring_to_list(args.range)
                rangedfn = 'animate.tmp.txt'
                traj = matio.load(sol_out)[li, :]
                matio.savetxt(rangedfn, traj)
                util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, rangedfn, '0.5'])
            else:
                util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, sol_out, '0.5'])
        else:
            util.warn("[tools.animate] Could not locate solution file {}".format(sol_out))

def vistouchdisp(args):
    ws = util.Workspace(args.dir)
    candidate_file = ws.local_ws(util.KEY_CANDIDATE_FILE)
    Qs = matio.load(candidate_file)['OMPL_CANDIDATES']
    nq = Qs.shape[0]
    qi_str = util.padded(args.key_id, nq)
    clearance_fn = ws.local_ws(util.PREP_KEY_CAN_SCRATCH, 'unitary_clearance_from_keycan-{}.npz'.format(qi_str))
    d = matio.load(clearance_fn)
    ukeys = d['TOUCH_V']
    fromv = d['FROM_V']
    import pyosr
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize

    diff = pyosr.multi_differential(fromv, ukeys, with_se3=False)
    trs = diff[:,0:3]
    # trs = normalize(trs)
    print(trs.shape)
    aas = diff[:,3:6]
    # aas = normalize(aas)
    fig1 = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.scatter3D(trs[:,0], trs[:,1], trs[:,2])
    fig2 = plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.scatter3D(aas[:,0], aas[:,1], aas[:,2])
    plt.show()

def blender_animate(args):
    # assert False
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    if not args.puzzle_name:
        util.fatal("[blender_animate] --puzzle_name is required to render")
        util.ack("[blender_animate] Avaliable puzzles:")
        for _, name in ws.test_puzzle_generator():
            util.ack(f"[blender_animate] {name}")
        return
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        cfg, config = parse_ompl.parse_simple(puzzle_fn)
        fl = FileLocations(args, ws, puzzle_name)

        calls = ['blender']
        if args.background:
            calls += ['-b']
        if args.condor_generate:
            calls += ['-t', '1'] # Single thread
        calls += ['-P', 'SolVis.py', '--']

        """
        Common
        """
        if args.saveas:
            calls += ['--saveas', args.saveas]
        if args.camera_origin is not None:
            calls += ['--camera_origin'] + ["{:.17f}".format(e) for e in args.camera_origin]
        if args.camera_lookat is not None:
            calls += ['--camera_lookat'] + ["{:.17f}".format(e) for e in args.camera_lookat]
        if args.camera_up is not None:
            calls += ['--camera_up'] + ["{:.17f}".format(e) for e in args.camera_up]
        if args.quit or args.background:
            calls += ['--quit']
        if args.cuda:
            calls += ['--cuda']
        if args.animation_single_frame is not None:
            assert not args.condor_generate
            calls += ['--animation_single_frame', str(args.animation_single_frame)]

        """
        SolVis specific
        """
        in_path = fl.unit_out_fn if args.use_unoptimized else fl.sim_out_fn
        vanilla_path = fl.vanilla_out_fn
        if not os.path.isfile(in_path):
            if not args.use_unoptimized:
                util.log(f'[blender_animate] The optimized solution file {in_path} does not exist')
                util.log(f'[blender_animate] Trying to run the path optimizer')
                o_args = TmpDriverArgs()
                o_args.current_trial = args.current_trial
                o_args.days = 0.01
                o_args.dir = args.dir
                simplify(o_args)
                if not os.path.isfile(in_path):
                    util.warn(f'[blender_animate] The path optimizer cannot get simplified solution')
                    continue
            else:
                util.warn(f'[blender_animate] The solution file {in_path} does not exist')
                continue
        uw = util.create_unit_world(puzzle_fn)
        if args.subtask == 'keyconf':
            ompl_q = matio.load(fl.get_assembled_raw_key_fn(trial=fl.trial))['KEYQ_OMPL']
            van_q = uw.translate_ompl_to_vanilla(ompl_q)
        elif args.subtask == 'keyconf_in_use':
            path_unit_q = matio.load(in_path)
            path_ompl_q = uw.translate_unit_to_ompl(path_unit_q)
            allkey_ompl_q = matio.load(fl.get_assembled_raw_key_fn(trial=fl.trial))['KEYQ_OMPL']
            key_ompl_q = allkey_ompl_q[2:,:] # Remove Initial state and goal state
            import pyosr
            ompl_q = []
            for key in key_ompl_q:
                dist = pyosr.multi_distance(key, path_ompl_q)
                if dist.min() < 1e-3:
                    ompl_q.append(key)
            ompl_q = np.array([allkey_ompl_q[0]] + ompl_q + [allkey_ompl_q[1]])
            print(f"keyconf_in_use shape {ompl_q.shape}")
            van_q = uw.translate_ompl_to_vanilla(ompl_q)
        else:
            unit_q = matio.load(in_path)
            van_q = uw.translate_ompl_to_vanilla(uw.translate_unit_to_ompl(unit_q))
        matio.savetxt(vanilla_path, van_q)
        oc = uw.get_ompl_center()
        ocstr = ["{:.17f}".format(oc[i]) for i in range(3)]
        DEBUG = False
        if DEBUG:
            import pyosr
            tau = 0.7507349475305162
            unit = pyosr.interpolate(unit_q[44], unit_q[45], tau).reshape((1,7))
            van = uw.translate_ompl_to_vanilla(uw.translate_unit_to_ompl(unit))
            print(f'Interpolate then translate: {van}')
            van2 = pyosr.interpolate(van_q[44], van_q[45], tau).reshape((1,7))
            print(f'Translate then interpolate: {van2}')
            ompl_q = uw.translate_unit_to_ompl(unit_q)
            van3 = pyosr.interpolate(ompl_q[44], ompl_q[45], tau).reshape((1,7))
            van3 = uw.translate_ompl_to_vanilla(van3)
            print(f'OMPL interpolate: {van3}')

        calls += ['--env', cfg.env_fn, '--rob', cfg.rob_fn, '--qpath', vanilla_path, '--O'] + ocstr
        if args.subtask == 'keyconf' or args.subtask == 'keyconf_in_use':
            calls += ['--discrete_points']
        if args.flat_env:
            calls += ['--flat_env']
        if args.mod_weighted_normal:
            calls += ['--mod_weighted_normal'] + args.mod_weighted_normal
        if args.floor_origin is not None:
            calls += ['--floor_origin'] + ["{:.17f}".format(e) for e in args.floor_origin]
        if args.animation_floor_origin is not None:
            calls += ['--animation_floor_origin'] + ["{:.17f}".format(e) for e in args.animation_floor_origin]
        if args.floor_euler is not None:
            calls += ['--floor_euler'] + ["{:.17f}".format(e) for e in args.floor_euler]
        if args.camera_from_bottom:
            calls += ['--camera_from_bottom']
        if args.light_panel_origin is not None:
            calls += ['--light_panel_origin'] + ["{:.17f}".format(e) for e in args.light_panel_origin]
        if args.light_panel_lookat is not None:
            calls += ['--light_panel_lookat'] + ["{:.17f}".format(e) for e in args.light_panel_lookat]
        if args.light_panel_up is not None:
            calls += ['--light_panel_up'] + ["{:.17f}".format(e) for e in args.light_panel_up]
        if args.light_auto:
            calls += ['--light_auto']
        if args.save_image:
            calls += ['--save_image', args.save_image]
        if args.save_animation_dir:
            calls += ['--save_animation_dir', args.save_animation_dir]
        if args.enable_animation_preview:
            calls += ['--enable_animation_preview']
        if args.enable_animation_overwrite:
            calls += ['--enable_animation_overwrite']
        if args.preview:
            calls += ['--preview']
        if args.image_frame is not None:
            calls += ['--image_frame', str(args.image_frame)]
        if args.animation_start >= 0:
            calls += ['--animation_start', str(args.animation_start)]
        if args.animation_end >= 0:
            calls += ['--animation_end', str(args.animation_end)]
        if args.condor_generate:
            assert args.save_animation_dir
            scratch = args.save_animation_dir
            end = args.animation_end if args.animation_end >= 0 else 1440
            from . import condor
            calls += ['--animation_single_frame', '$(Process)']
            calls += args.additional_arguments
            os.makedirs(scratch, exist_ok=True)
            subfile= condor.local_submit(ws=ws,
                                         xfile='/usr/bin/env',
                                         iodir_rel='',
                                         arguments=['blender'] + calls[1:],
                                         instances=end,
                                         local_scratch=scratch,
                                         wait=False,
                                         dryrun=False)
            util.ack(f'[tool blender] write condor submission file to {subfile}')
        else:
            calls += args.additional_arguments
            util.shell(calls)

def blender_texture(args):
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        cfg, config = parse_ompl.parse_simple(puzzle_fn)
        fl = FileLocations(args, ws, puzzle_name)
        calls = ['blender']
        if args.background:
            calls += ['-b']
        if args.condor_generate:
            calls += ['-t', '1'] # Single thread
        calls += ['-P', 'BTexVis.py', '--']

        """
        Common
        """
        if args.saveas:
            calls += ['--saveas', args.saveas]
        if args.camera_origin is not None:
            calls += ['--camera_origin'] + ["{:.17f}".format(e) for e in args.camera_origin]
        if args.camera_lookat is not None:
            calls += ['--camera_lookat'] + ["{:.17f}".format(e) for e in args.camera_lookat]
        if args.camera_up is not None:
            calls += ['--camera_up'] + ["{:.17f}".format(e) for e in args.camera_up]
        if args.quit or args.background:
            calls += ['--quit']
        if args.cuda:
            calls += ['--cuda']
        if args.animation_single_frame is not None:
            assert not args.condor_generate
            calls += ['--animation_single_frame', str(args.animation_single_frame)]
        if args.flat_env:
            calls += ['--flat']

        """
        BTexVis specific
        """
        for geo_type, geo_fn in zip(['env', 'rob'], [cfg.env_fn, cfg.rob_fn]):
            if geo_type == 'env':
                rx = args.env_camera_rotation_axis
            else:
                rx = args.rob_camera_rotation_axis
            for atex_id, atex in fl.get_atex_file_gen(puzzle_fn, geo_type):
                if args.netid is not None and atex_id != args.netid:
                    continue
                pc_calls = list(calls)
               # ['--camera_rotation_axis'] + ["{:.17f}".format(e) for e in rx] +
                if args.save_texture_visualization_dir:
                    pc_calls += ['--save_animation_dir', f'{args.save_texture_visualization_dir}/{geo_type}#{atex_id}']
                util.shell(pc_calls + [geo_fn, atex] )
                # break # Debug

def blender_entrance(args):
    print(args)
    if args.subtask == 'texture':
        blender_texture(args)
    elif args.subtask == 'keyconf':
        blender_animate(args)
    elif args.subtask == 'keyconf_in_use':
        args.use_unoptimized = True
        blender_animate(args)
    else:
        blender_animate(args)

def simplify(args):
    from . import solve2
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        fl = FileLocations(args, ws, puzzle_name)
        fl.update_scheme('cmb')
        unit_path = fl.unit_out_fn
        util.log(f'Simplifiying trajectory from {unit_path}')
        unit_q = matio.load(unit_path)
        uw = util.create_unit_world(puzzle_fn)
        ompl_q = uw.translate_unit_to_ompl(unit_q)
        driver = solve2.create_driver(puzzle_fn)
        sim_q = driver.optimize(ompl_q, args.days)
        unit_sim_q = uw.translate_ompl_to_unit(sim_q)
        matio.savetxt(fl.sim_out_fn, unit_sim_q)
        util.log(f'Simplified trajectory written to {fl.sim_out_fn}')

def dump_training_data(args):
    ws = util.Workspace(args.dir)
    from . import hg_launcher
    params = hg_launcher.create_config_from_profile(args.nn_profile)
    all_omplcfgs = []
    #for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
    for puzzle_fn, puzzle_name in ws.training_puzzle_generator():
        all_omplcfgs.append(puzzle_fn)
        break
    params['all_ompl_configs'] = all_omplcfgs
    params['what_to_render'] = args.geo_type
    params['checkpoint_dir'] = ws.checkpoint_dir(args.geo_type) + '/'
    # params['enable_augmentation'] = False
    """
    params['suppress_hot'] = 0.0
    params['suppress_cold'] = 0.0
    params['red_noise'] = 0.9
    """
    params['suppress_hot'] = 0.25
    params['suppress_cold'] = 0.25
    params['red_noise'] = 0.25
    params['dataset_name'] = f'{ws.dir}.{args.geo_type}'
    from . import hg_datagen
    dataset = hg_datagen.create_dataset_from_params(params)
    gen = dataset._aux_generator(batch_size=16, stacks=2, normalize=True, sample_set='train')
    img_train, gt_train, _ = next(gen)
    assert 0 <= gt_train.any() <= 1.0, 'incorrect gt_train mag. max {np.max(gt_train)} min {np.max(gt_train)}'
    index = 0
    for img, gt in zip(img_train, gt_train):
        if dataset.gen_surface_normal:
            rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
            rgb[:, :, 0] = img[:,:,0]
            imsave(f'{args.out}/{index}-rgb.png', rgb)
            imsave(f'{args.out}/{index}-normal.png', img[:,:,1:4])
            imsave(f'{args.out}/{index}-dep.png', img[:,:,4])
            imsave(f'{args.out}/{index}-hm.png', gt[0])
        else:
            imsave(f'{args.out}/{index}-rgb.png', img[:,:,0:3])
            imsave(f'{args.out}/{index}-dep.png', img[:,:,3])
            imsave(f'{args.out}/{index}-hm.png', gt[0])
        index += 1

def debug(args):
    from . import parse_ompl
    for puzzle_fn in args.arguments:
        cfg, config = parse_ompl.parse_simple(puzzle_fn)
        p = pathlib.Path(puzzle_fn)
        infile = str(p.with_suffix(".ompl.txt"))
        insfile = str(p.with_suffix(".simplified.txt"))
        outfn = str(p.with_suffix(".npz"))
        if not isfile(infile):
            continue
        uw = util.create_unit_world(puzzle_fn)

        """
        ompl_q = matio.load(infile)
        if not isfile(insfile):
            from . import solve2
            util.ack(f'Simplifiying {puzzle_fn}')
            driver = solve2.create_driver(puzzle_fn)
            sim_q = driver.optimize(ompl_q, 0.01)
        else:
            sim_q = matio.load(insfile)
            sim_q = uw.translate_unit_to_ompl(sim_q)
        oc = uw.get_ompl_center()
        np.savez(outfn, OMPL_PATH=ompl_q, SIMPLIFIED_PATH=sim_q, REFERENCE_ORIGIN=oc)
        util.ack(f'save file to {outfn}')
        """
        d = matio.load(outfn)
        matio.savetxt(insfile, d['SIMPLIFIED_PATH'])
    return
    puzzle_fn = 'u2/claw-rightbv.dt.tcp/test/claw-rightbv.dt.tcp/puzzle.cfg'
    # keys = matio.load('condor.u2/claw-rightbv.dt.tcp/test/claw-rightbv.dt.tcp/', KEY='KEYQ_OMPL')
    keys = matio.load('u2/claw-rightbv.dt.tcp/test/claw-rightbv.dt.tcp/geometrik_geratio_keyconf-20.npz', key='KEYQ_OMPL')
    """
    uw = util.create_unit_world(puzzle_fn)
    print(uw.calculate_visibility_pair(keys[[989], :], False,
                                       keys[[990], :], False,
                                       uw.recommended_cres,
                                       enable_mt=False))
    """
    from . import se3solver
    driver_args = TmpDriverArgs()
    driver_args.puzzle = puzzle_fn
    driver_args.planner_id = se3solver.PLANNER_RDT
    driver_args.sampler_id = 0
    driver = se3solver.create_driver(driver_args)
    print(driver.validate_motion_pairs(keys[[989], :], keys[[990], :]))
    return
    print(args)
    return
    import pygeokey
    puzzle_fn = 'condor.uw/duet-g9/test/duet-g9/puzzle.cfg'
    env_fn = 'condor.uw/duet-g9/test/duet-g9/duet.dt.tcp.obj'
    rob_fn = 'condor.uw/duet-g9/test/duet-g9/knotted_ring.dt.tcp.obj'
    nrot = 256
    env_feat = matio.load('condor.uw/duet-g9/test/duet-g9/geometrik_notch_point_of_env-50.npz', key='NOTCH_POINT_AMBIENT')
    rob_feat = matio.load('condor.uw/duet-g9/test/duet-g9/geometrik_geratio_point_of_rob-50.npz', key='KEY_POINT_AMBIENT')
    # env_feat = matio.load('condor.uw/duet-g9/test/duet-g9/geometrik_geratio_point_of_env-50.npz', key='KEY_POINT_AMBIENT')
    # rob_feat = matio.load('condor.uw/duet-g9/test/duet-g9/geometrik_notch_point_of_rob-50.npz', key='NOTCH_POINT_AMBIENT')
    ks = pygeokey.KeySampler(env_fn, rob_fn)
    keys, keyid_env, keyid_rob = ks.get_all_key_configs(env_feat, rob_feat, nrot)
    util.log(f"keys {keys.shape} from {env_feat.shape} and {rob_feat.shape}")
    if keys.shape[0] == 0:
        util.log("No key configurations predicated, exiting")
        return
    uw = util.create_unit_world(puzzle_fn)
    ukeys = uw.translate_ompl_to_unit(keys)
    matio.savetxt('viskey.tmp.txt', ukeys)
    cmd = ['./vispath', env_fn, rob_fn, 'viskey.tmp.txt', '0.5']
    util.shell(cmd)

function_dict = {
        'read_roots' : read_roots,
        'write_roots' : write_roots,
        'visenvgt' : visenvgt,
        'visrobgt' : visrobgt,
        'vistraj' : vistraj,
        'plotstat' : plotstat,
        'visclearance' : visclearance,
        'visnnpred' : visnnpred,
        'visnnsample' : visnnsample,
        'viskey' : viskey,
        'visfeat' : visfeat,
        'visnotch' : visnotch,
        'vistouchv' : vistouchv,
        'vistouchdisp' : vistouchdisp,
        'animate' : animate,
        'blender' : blender_entrance,
        'simplify' : simplify,
        'dump_training_data' : dump_training_data,
        'debug' : debug,
}

def setup_parser(subparsers):
    sp = subparsers.add_parser('tools', help='Various Tools.')
    toolp = sp.add_subparsers(dest='tool_name', help='Name of Tool')

    p = toolp.add_parser('read_roots', help='Dump roots of the forest to text file')
    p.add_argument('--roots_key', help='NPZ file of roots', default='KEYQ_OMPL')
    p.add_argument('--to_vanilla', help='Store vanilla instead of unit configurations', action='store_true')
    p.add_argument('puzzle_fn', help='OMPL config')
    p.add_argument('roots', help='NPZ file of roots')
    p.add_argument('out', help='output txt file')

    p = toolp.add_parser('write_roots', help='Write unitary configurations as roots of the forest to text file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input_key', help='key of array that stores roots. Useful for non-text format', default=None)
    p.add_argument('--roots_key', help='NPZ key of roots', default='KEYQ_OMPL')
    p.add_argument('--current_trial', help='Trial of workspace', type=int, default=0)
    p.add_argument('--puzzle_name', help='Puzzle name in workspace', required=True)
    p.add_argument('--noig', help='Do not add initial state and goal state as Key 0 and Key 1', action='store_true')
    p.add_argument('roots', help='file of unitary configurations supported by matio')
    p.add_argument('dir', help='workspace')

    p = toolp.add_parser('visenvgt', help='Call vistexture to visualize the training texture')
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('visrobgt', help='Call vistexture to visualize the training texture')
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('vistraj', help='Call vispath to visualize the training trajectory')
    p.add_argument('--traj_id', help='Trajectory ID', default=0)
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('plotstat', help='Call matplotlib and vispath to show the statistics of the training trajectory')
    p.add_argument('--noupdate', help='Do not sync with remote host', action='store_true')
    p.add_argument('--top_k', help='Top K', type=int, default=None)
    p.add_argument('--offset', help='Specify which serie of stats to plot', type=int, default=0)
    p.add_argument('--trim', help='Trim the tail', type=int, default=0)
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('visclearance', help='Show the colliding configuration used to calculate the clearance')
    p.add_argument('dir', help='Workspace directory')
    p.add_argument('--traj_id', help='Specify the trajectory id', type=int, default=0)
    p.add_argument('--point_id', help='Specify the point in the trajectory', type=int, default=0)

    p = toolp.add_parser('visnnpred', help='Call vistexture to visualize the prediction results')
    p.add_argument('--current_trial', help='Trial to predict the keyconf', type=int, default=0)
    p.add_argument('--puzzle_name', help='Only show one specific puzzle', default='')
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('visnnsample', help='Call vistexture to visualize the prediction results')
    p.add_argument('--puzzle_name', help='Only show one specific puzzle', default='')
    p.add_argument('--update', help='Only show one specific puzzle', action='store_true')
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('viskey', help='Use vispath to visualize the key configuration')
    p.add_argument('--scheme', help='Select which set of key points to visualize', default=None, choices=KEY_PRED_SCHEMES)
    p.add_argument('--current_trial', help='Trial to predict the keyconf', type=int, default=None)
    p.add_argument('--range', help='Range of key confs, e.g. 1,2,3,4-7,11', default='')
    p.add_argument('--puzzle_name', help='Only show one specific puzzle', default='')
    p.add_argument('--unscreened', help='Show the unscreened key configurations', action='store_true')
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('visfeat', help='Use vispath to visualize the key configuration')
    p.add_argument('--current_trial', help='Trial to predict the keyconf', type=int, default=None)
    p.add_argument('--range', help='Range of key confs, e.g. 1,2,3,4-7,11', default='')
    p.add_argument('--puzzle_name', help='Only show one specific puzzle', default='')
    p.add_argument('--scheme', help='Select which set of key points to visualize', default=None, choices=FEAT_PRED_SCHEMES)
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('visnotch', help='Visualize Notch from geometric hueristics')
    p.add_argument('--current_trial', help='Trial to predict the keyconf', type=int, default=None)
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('vistouchv', help='Visualize the touch configurations in clearance estimation')
    p.add_argument('--key_id', help='Key ID in KeyCan.npz. Top K can be found at KEY.npz:_TOP_K', type=int, required=True)
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('vistouchdisp', help='Visualize the displacement of touch configurations from clearance estimation')
    p.add_argument('--key_id', help='Key ID in KeyCan.npz. Top K can be found at KEY.npz:_TOP_K', type=int, required=True)
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('animate', help='Show the animation of solution with vispath')
    p.add_argument('--current_trial', help='Trial to predict the keyconf', type=int, default=None)
    p.add_argument('--range', help='Only use a subset of the path, format example: 1,2,3,4-7,11', default='')
    p.add_argument('--puzzle_name', help='Only show one specific puzzle', default='')
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('blender', help='Invoke blender to render the trajectory')
    p.add_argument('--subtask', help='Which task to perform',
                   choices=['texture', 'keyconf', 'keyconf_in_use', 'solution_trajectory'],
                   default='solution_trajectory')
    p.add_argument('--netid', type=int, default=None)
    p.add_argument('--current_trial', help='Trial that has the solution trajectory', type=int, default=None)
    p.add_argument('--puzzle_name', help='Puzzle selection. Will exit after displaying all puzzle names if not present', default='')
    p.add_argument('--scheme', help='Scheme selection', choices=KEY_PRED_SCHEMES, default='cmb')
    p.add_argument('--condor_generate', help='Generate HTCondor submission file, instead of running locally. --save_animation_dir is mandatory and used as output and scratch dir', action='store_true')
    p.add_argument('--use_unoptimized', action='store_true')
    p.add_argument('--camera_origin', help='Origin of camera', type=float, nargs=3, default=None)
    p.add_argument('--camera_lookat', help='Point to Look At of camera', type=float, nargs=3, default=None)
    p.add_argument('--camera_up', help='Up direction of camera', type=float, nargs=3, default=None)
    p.add_argument('--camera_from_bottom', help='flip_camera w.r.t. the lookat and up direction. This also make a transparent floor', action='store_true')
    p.add_argument('--env_camera_rotation_axis', type=float, nargs=3, default=None)
    p.add_argument('--rob_camera_rotation_axis', type=float, nargs=3, default=None)
    p.add_argument('--floor_origin', help='Center of the floor', type=float, nargs=3, default=None)
    p.add_argument('--floor_euler', help='Rotate the floor with euler angle', type=float, nargs=3, default=None)
    p.add_argument('--light_panel_origin', help='Origin of light_panel', type=float, nargs=3, default=None)
    p.add_argument('--light_panel_lookat', help='Point to Look At of light_panel', type=float, nargs=3, default=None)
    p.add_argument('--light_panel_up', help='Up direction of light_panel', type=float, nargs=3, default=None)
    p.add_argument('--light_auto', help='Set the light configuration automatically', action='store_true')
    p.add_argument('--flat_env', help='Flat shading', action='store_true')
    p.add_argument('--mod_weighted_normal', help='Add modifier "Weighted Normal"', choices=['env', 'rob'], nargs='*', default=[])
    p.add_argument('--image_frame', help='Still Image Frame. Use in conjuction with --save_image', type=int, default=None)
    p.add_argument('--animation_single_frame', help='Render single frame of animation. Use in conjuction with --save_animation_dir. Override animation_end.', type=int, default=None)
    p.add_argument('--animation_start', help='Start frame of animation', type=int, default=-1)
    p.add_argument('--animation_end', help='End frame of animation', type=int, default=-1)
    p.add_argument('--animation_floor_origin', help='Center of the floor when rendering the animation',
                                     type=float, nargs=3, default=[0, 0, -40])
    p.add_argument('--saveas', help='Save the Blender file as', default='')
    p.add_argument('--save_image', help='Save the Rendered image as', default='')
    p.add_argument('--save_animation_dir', help='Save the Rendered animation sequence image to', default='')
    p.add_argument('--save_texture_visualization_dir', help='Save the Rendered Texture Heatmap to', default='')
    p.add_argument('--enable_animation_preview', action='store_true')
    p.add_argument('--enable_animation_overwrite', action='store_true')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--preview', action='store_true')
    p.add_argument('--quit', help='Quit without running blender', action='store_true')
    p.add_argument('--background', help='Run blender in background. Implies --quit', action='store_true')
    p.add_argument('--dir', help='Workspace directory', required=True)
    p.add_argument('additional_arguments', help='Addtional arguments passed to the underlying script', nargs="*")

    p = toolp.add_parser('dump_training_data', help='Dump the training data to output file')
    p.add_argument('--puzzle_name', help='Training Puzzle selection. The name of the default puzzle is "train".', default='')
    p.add_argument('--geo_type', help='Type of the geometry (env/rob)', default='rob')
    p.add_argument('--nn_profile', help='NN profile', default='256hg+normal')
    p.add_argument('dir', help='Workspace directory')
    p.add_argument('out', help='Output directory')

    p = toolp.add_parser('simplify', help='Simplify the trajectory')
    p.add_argument('--current_trial', help='Trial to simplify', type=int, default=0)
    p.add_argument('--days', help='Time limit of optimization', type=float, default=0.01)
    p.add_argument('dir', help='Workspace directory')

    p = toolp.add_parser('debug', help='Temporary debugging code. Eveything should be hardcoded')
    p.add_argument('arguments', help='Custom arguments', nargs='*')

def run(args):
    function_dict[args.tool_name](args)
