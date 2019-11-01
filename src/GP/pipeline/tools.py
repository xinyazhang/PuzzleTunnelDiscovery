#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isdir, isfile
import copy
import pathlib
import csv
import numpy as np
import argparse
from imageio import imwrite as imsave

from . import util
from . import matio
from . import condor
from . import parse_ompl
from . import atlas
from . import partt
from .file_locations import FEAT_PRED_SCHEMES, RAW_KEY_PRED_SCHEMES, KEY_PRED_SCHEMES, SCHEME_TO_FMT, SCHEME_FEAT_NPZ_KEY, FileLocations

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
        env_npz = ws.atex_prediction_file(puzzle_fn, 'env', trial_override=args.current_trial)
        util.shell(['./atex2green.py', env_npz])
        rob_npz = ws.atex_prediction_file(puzzle_fn, 'rob', trial_override=args.current_trial)
        util.shell(['./atex2green.py', rob_npz])
        p = pathlib.Path(env_npz)
        util.shell(['./vistexture', cfg.env_fn, str(p.with_suffix('.png'))])
        p = pathlib.Path(rob_npz)
        util.shell(['./vistexture', cfg.rob_fn, str(p.with_suffix('.png'))])

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
            ws.config.getint('DEFAULT', 'CondorQuota') * 2,
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

def _dic_add(dic, key, v):
    if v is None:
        return
    if key in dic:
        dic[key].append(v)
    else:
        dic[key] = [v]

def _print_detail_header(writer):
    writer.writerow(['Puzzle Name',
                     'Trial ID',
                     'Method',
                     'Key Points (ROB)',
                     'Key Points (ENV)',
                     'Number of Roots',
                     'PDS size',
                     'Solved (Y/N)',
                    ])


def _print_detail(puzzle_name, stat_dic, writer):
    for i in range(len(stat_dic['trial_id'])):
        writer.writerow([puzzle_name,
                         stat_dic['trial_id'][i],
                         stat_dic['puzzle_method'][i],
                         '{}'.format(stat_dic['puzzle_kps_env'][i]),
                         '{}'.format(stat_dic['puzzle_kps_rob'][i]),
                         # stat_dic['puzzle_rot'][i],
                         stat_dic['puzzle_roots'][i],
                         stat_dic['puzzle_pds'][i],
                         stat_dic['puzzle_success'][i]])

def _print_stat_header(writer):
    writer.writerow(['Puzzle Name',
                     'Trial IDs',
                     'Method',
                     'Mean Key Points (ROB)',
                     'Stdev of Key Points (ROB)',
                     'Mean Key Points (ENV)',
                     'Stdev of Key Points (ENV)',
                     'Mean Number of Roots',
                     'Mean PDS size',
                     'Solved/Total',
                     'Solved/Total (WithBT)',
                     'Solved/Total (KNN Ver. 3)',
                    ])

def _print_stat(puzzle_name, stat_dic, writer):
    writer.writerow([puzzle_name, stat_dic['trial_range'],
                     stat_dic['puzzle_method'][0] if all(elem == stat_dic['puzzle_method'][0] for elem in stat_dic['puzzle_method']) else '*MIXED*',
                     '{}'.format(np.mean(stat_dic['puzzle_kps_env'])),
                     '{}'.format(np.std(stat_dic['puzzle_kps_env'])),
                     '{}'.format(np.mean(stat_dic['puzzle_kps_rob'])),
                     '{}'.format(np.std(stat_dic['puzzle_kps_rob'])),
                     # '{}'.format(np.mean(stat_dic['puzzle_rot'])),
                     '{}'.format(np.mean(stat_dic['puzzle_roots'])),
                     '{}'.format(np.mean(stat_dic['puzzle_pds'])),
                     '{}/{}'.format(np.sum(stat_dic['puzzle_success_int']),
                                    len(stat_dic['puzzle_success'])),
                     '{}/{}'.format(np.sum(stat_dic['puzzle_withbt_success_int']),
                                    len(stat_dic['puzzle_withbt_success'])),
                     '{}/{}'.format(np.sum(stat_dic['puzzle_knn3_success_int']),
                                    len(stat_dic['puzzle_knn3_success_int']))
                    ])

def conclude(args):
    f = open(args.out, 'w')
    writer = csv.writer(f)
    if args.type == 'detail':
        _print_detail_header(writer)
    elif args.type == 'stat':
        _print_stat_header(writer)
    for ws_dir in args.dirs:
        ws = util.Workspace(ws_dir)
        trial_list = util.rangestring_to_list(args.trial_range)
        for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
            if args.puzzle_name and puzzle_name != args.puzzle_name:
                continue
            cfg, config = parse_ompl.parse_simple(puzzle_fn)
            stat_dic = {}
            for trial in trial_list:
                ws.current_trial = trial
                pds_fn = ws.local_ws(util.SOLVER_SCRATCH,
                                     puzzle_name,
                                     util.PDS_SUBDIR,
                                     '{}.npz'.format(trial))
                if not os.path.exists(pds_fn):
                    continue
                kp_env_fn = ws.keypoint_prediction_file(puzzle_name, 'env')
                kp_rob_fn = ws.keypoint_prediction_file(puzzle_name, 'rob')
                overkp_fn = ws.oversampled_keyconf_prediction_file(puzzle_name)
                puzzle_roots_from_gk = None
                puzzle_roots_from_nn_oversampled = None
                puzzle_roots_from_nn = None
                if os.path.exists(kp_rob_fn) and os.path.exists(kp_env_fn):
                    puzzle_method = 'GK'
                    puzzle_rot = ws.config.getint('GeometriK', 'KeyConfigRotations')
                    d_env = matio.load(kp_env_fn)
                    d_rob = matio.load(kp_rob_fn)
                    puzzle_kps_env = util.access_keypoints(d_env, 'env').shape[0]
                    puzzle_kps_rob = util.access_keypoints(d_rob, 'rob').shape[0]
                    if os.path.exists(overkp_fn):
                        puzzle_method = 'GK+NN'
                        puzzle_roots_from_nn = matio.load(overkp_fn)['KEYQ_OMPL'].shape[0]
                        puzzle_roots_from_gk = None
                        FMT = util.GEOMETRIK_KEY_PREDICTION_FMT
                        kfn = ws.keyconf_file_from_fmt(puzzle_name, FMT)
                        puzzle_roots_from_gk = matio.load(kfn)['KEYQ_OMPL'].shape[0]
                        kfn = ws.keyconf_prediction_file(puzzle_name)
                        puzzle_roots_from_nn = matio.load(kfn)['KEYQ_OMPL'].shape[0]
                else:
                    puzzle_method = 'NN'
                    puzzle_rot = ws.config.getint('Prediction', 'NumberOfRotations')
                    puzzle_kps_env = -1
                    puzzle_kps_rob = -1
                kq_fn = ws.screened_keyconf_prediction_file(puzzle_name)
                puzzle_roots = matio.load(kq_fn)['KEYQ_OMPL'].shape[0]
                puzzle_pds = matio.load(pds_fn)['Q'].shape[0]
                sol_fn = ws.solution_file(puzzle_name, type_name='unit')
                if os.path.exists(sol_fn):
                    puzzle_success = 'Y'
                else:
                    puzzle_success = 'N'
                sol_fn = ws.solution_file(puzzle_name, type_name='withbt-unit')
                if os.path.exists(sol_fn):
                    puzzle_withbt_success = 'Y'
                else:
                    puzzle_withbt_success = 'N'
                sol_fn = ws.solution_file(puzzle_name, type_name='pairwise_knn-unit')
                if os.path.exists(sol_fn):
                    puzzle_knn3_success = 'Y'
                else:
                    puzzle_knn3_success = 'N'
                _dic_add(stat_dic, 'trial_id', trial)
                _dic_add(stat_dic, 'puzzle_method', puzzle_method)
                _dic_add(stat_dic, 'puzzle_kps_env', puzzle_kps_env)
                _dic_add(stat_dic, 'puzzle_kps_rob', puzzle_kps_rob)
                _dic_add(stat_dic, 'puzzle_rot', puzzle_rot)
                _dic_add(stat_dic, 'puzzle_roots', puzzle_roots)
                _dic_add(stat_dic, 'puzzle_gk_roots', puzzle_roots_from_gk)
                _dic_add(stat_dic, 'puzzle_nn_roots_oversampled', puzzle_roots_from_nn_oversampled)
                _dic_add(stat_dic, 'puzzle_nn_roots', puzzle_roots_from_nn)
                _dic_add(stat_dic, 'puzzle_pds', puzzle_pds)
                _dic_add(stat_dic, 'puzzle_success', puzzle_success)
                _dic_add(stat_dic, 'puzzle_success_int', 1 if puzzle_success == 'Y' else 0)
                _dic_add(stat_dic, 'puzzle_withbt_success', puzzle_withbt_success)
                _dic_add(stat_dic, 'puzzle_withbt_success_int', 1 if puzzle_withbt_success == 'Y' else 0)
                _dic_add(stat_dic, 'puzzle_knn3_success', puzzle_knn3_success)
                _dic_add(stat_dic, 'puzzle_knn3_success_int', 1 if puzzle_knn3_success == 'Y' else 0)
            stat_dic['trial_range'] = args.trial_range
            if 'puzzle_success' not in stat_dic:
                util.warn('workspace {} has not solution data for puzzle {}. No corresponding information will be printed'.format(ws_dir, puzzle_name))
                continue
            if args.type == 'detail':
                _print_detail(puzzle_name, stat_dic, writer)
            elif args.type == 'stat':
                _print_stat(puzzle_name, stat_dic, writer)
    f.close()

def _get_detailed_rows(grand_dict):
    yield_header = True
    for trial in grand_dict:
        for puzzle_name in grand_dict[trial]:
            data = [cost for _,cost in grand_dict[trial][puzzle_name]]
            if yield_header:
                yield ['Puzzle Name', 'Trial ID'] + [stage_name for stage_name,_ in grand_dict[trial][puzzle_name]]
                yield_header = False
            yield [puzzle_name, trial] + data

def _get_stat_rows(grand_dict):
    pass

def _get_rows(grand_dict, args):
    if args.type == 'detail':
        yield from _get_detailed_rows(grand_dict)
    elif args.type == 'stat':
        yield from _get_stat_rows(grand_dict)

def _detect_single_puzzle(ws):
    # Deal with single puzzle workspace
    npuzzle = 0
    last_puzzle_name = None
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        npuzzle += 1
        last_puzzle_name = puzzle_name
        if npuzzle > 1:
            break
    if npuzzle == 1:
        return last_puzzle_name
    return None

def _parse_log(logfn, single_puzzle):
    ret_dic = {}
    breaker = ' cost '
    _RIGHT_TO_WRONG = {
            'forest_rdt_withbt'     : 'forest_rdt',
            'forest_edges_withbt'   : 'forest_edges',
            'connect_forest_withbt' : 'connect_forest'
    }

    _WRONG_TO_RIGHT = {
        'forest_rdt'     : 'forest_rdt_withbt',
        'forest_edges'   : 'forest_edges_withbt',
        'connect_forest' : 'connect_forest_withbt'
    }

    def _update_ret_dic(list_of_tuples, stage_name, cost_str):
        for i in range(len(list_of_tuples)):
            k,v = list_of_tuples[i]
            if k == stage_name:
                list_of_tuples[i] = (stage_name, cost_str)
                return
        list_of_tuples.append((stage_name, cost_str))

    with open(logfn, 'r') as f:
        '''
        We had a bug in recording forest_rdt_withbt, forest_edges_withbt, connect_forest_withbt
        Hence we are going to introduce a context-dependent parser to fix this
        '''
        fixing = ''
        fixed_as = ''
        for line in f:
            loc = line.find(breaker)
            if loc < 0:
                continue
            cost_str = line[loc+len(breaker):].strip()
            line = line.replace('[', ' ')
            line = line.replace(']', ' ')
            split = line.split()
            stage_name = split[0]
            if split[1] == 'cost': # old format, '[puzzle name]' not exist
                puzzle_name = '*'
            else:
                puzzle_name = split[1]
            if single_puzzle is not None:
                puzzle_name = single_puzzle if puzzle_name == '*' else puzzle_name
            if puzzle_name not in ret_dic:
                ret_dic[puzzle_name] = []
            if 'puzzle_name' == '*' and stage_name in _RIGHT_TO_WRONG:
                if split[2] == 'starting':
                    fixing = _RIGHT_TO_WRONG[stage_name]
                    fixed_as = stage_name
                elif split[2] == 'finished':
                    fixing = ''
                    fixed_as = ''
            else:
                if fixing == stage_name:
                    stage_name = fixed_as
            # print("{} {}".format(stage_name, cost_str))
            _update_ret_dic(ret_dic[puzzle_name], stage_name, cost_str)
    return ret_dic

def breakdown(args):
    grand_dict = {}
    trial_list = util.rangestring_to_list(args.trial_range)

    for ws_dir in args.dirs:
        ws = util.Workspace(ws_dir)
        single_puzzle = _detect_single_puzzle(ws)
        for trial in trial_list:
            if trial not in grand_dict:
                grand_dict[trial] = {}
            logfn = ws.local_ws(util.PERFORMANCE_LOG_DIR, 'log.{}'.format(trial))
            if not os.path.exists(logfn):
                util.log("{} does not exist, skipping".format(logfn))
                continue
            grand_dict[trial].update(_parse_log(logfn, single_puzzle))

    f = open(args.out, 'w')
    writer = csv.writer(f)
    for row in _get_rows(grand_dict, args):
        writer.writerow(row)
    f.close()

_CONDOR_PPSTAGE_TO_DIR = {
        'find_trajectory': util.PREP_TRAJECTORY_SCRATCH,
        'estimate_clearance_volume': util.PREP_KEY_CAN_SCRATCH,
        'sample_touch' : util.PREP_TOUCH_SCRATCH,
        'isect_geometry' : util.PREP_ISECT_SCRATCH,
        'uvproject' : util.UV_DIR,
}

def estimate_keyconf_clearance_dir(puzzle_name, current_trial):
    return join(util.SOLVER_SCRATCH, puzzle_name, util.KEYCONF_CLEARANCE_DIR, str(current_trial))

def screen_keyconf_dir(puzzle_name, current_trial):
    return join(util.SOLVER_SCRATCH, puzzle_name, 'screen-{}'.format(current_trial))

def sample_pds_dir(puzzle_name, current_trial):
    return join(util.SOLVER_SCRATCH, puzzle_name, util.PDS_SUBDIR, 'bloom-{}'.format(current_trial))

def forest_rdt_dir(puzzle_name, current_trial):
    return join(util.SOLVER_SCRATCH, puzzle_name, 'trial-{}'.format(current_trial))

def forest_rdt_withbt_dir(puzzle_name, current_trial):
    return join(util.SOLVER_SCRATCH, puzzle_name, 'withbt-trial-{}'.format(current_trial))

def knn3_dir(puzzle_name, current_trial):
    return join(util.SOLVER_SCRATCH, puzzle_name, 'pairwise_knn-{}'.format(current_trial))

_CONDOR_SOLSTAGE_TO_DIR = {
        'estimate_keyconf_clearance' : estimate_keyconf_clearance_dir,
        'screen_keyconf' : screen_keyconf_dir,
        'sample_pds' : sample_pds_dir,
        'forest_rdt' : forest_rdt_dir,
        'forest_rdt_withbt' : forest_rdt_withbt_dir,
        'knn3' : knn3_dir,
}

def condor_ppbreakdown(args):
    grand_dict = {}
    trial_list = util.rangestring_to_list(args.trial_range)
    for ws_dir in args.dirs:
        ws = util.Workspace(ws_dir)
        pp_dict = {}
        for k,v in _CONDOR_PPSTAGE_TO_DIR.items():
            lt = condor.query_last_cputime(ws, v)
            if lt is not None:
                pp_dict[k] = lt
        ws_basename = os.path.basename(ws_dir)
        grand_dict[ws_basename] = pp_dict
    def _get_rows():
        first = True
        keys = []
        for ent, dic in grand_dict.items():
            if first:
                ty = ['Workspace']
                for k,v in dic.items():
                    keys += [k]
                yield ty + keys
                first = False
            ty = [ent]
            for k in keys:
                ty.append(dic[k])
            yield ty
    with open(args.out, 'w') as f:
        writer = csv.writer(f)
        for row in _get_rows():
            writer.writerow(row)

def condor_breakdown(args):
    grand_dict = {}
    trial_list = util.rangestring_to_list(args.trial_range)

    for ws_dir in args.dirs:
        ws = util.Workspace(ws_dir)
        for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
            if puzzle_name not in grand_dict:
                grand_dict[puzzle_name] = {}
            pp_dict = grand_dict[puzzle_name]
            for trial in trial_list:
                if trial not in grand_dict:
                    pp_dict[trial] = {}
                for k,v in _CONDOR_SOLSTAGE_TO_DIR.items():
                    rel = v(puzzle_name, trial)
                    lt = condor.query_last_cputime(ws, rel)
                    if lt is not None:
                        pp_dict[trial][k] = lt
    print(grand_dict)
    def _get_rows():
        first = True
        keys = []
        for puzzle_name, dic in grand_dict.items():
            print(puzzle_name)
            print(dic)
            if first:
                ty = ['Puzzle', 'Trial']
                for trial, trialdata in dic.items():
                    for k,v in trialdata.items():
                        keys += [k]
                    yield ty + keys
                    first = False
                    break
            for trial, trialdata in dic.items():
                ty = [puzzle_name, trial]
                for k in keys:
                    if k in trialdata:
                        ty.append(trialdata[k])
                    else:
                        ty.append('N/A')
                yield ty
    with open(args.out, 'w') as f:
        writer = csv.writer(f)
        for row in _get_rows():
            writer.writerow(row)

def blender_animate(args):
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
        unit_path = ws.solution_file(puzzle_name, type_name='unit')
        vanilla_path = ws.solution_file(puzzle_name, type_name='vanilla')
        if not os.path.isfile(unit_path):
            util.warn(f'[blender_animate] The solution file {unit_path} does not exist')
            continue
        uw = util.create_unit_world(puzzle_fn)
        unit_q = matio.load(unit_path)
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
        util.shell(['blender', '-P', 'SolVis.py', '--', cfg.env_fn, cfg.rob_fn, vanilla_path, args.outdir, '--O'] + ocstr)

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
        'conclude' : conclude,
        'breakdown' : breakdown,
        'condor_breakdown' : condor_breakdown,
        'condor_ppbreakdown' : condor_ppbreakdown,
        'blender' : blender_animate,
        'dump_training_data' : dump_training_data,
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

    p = toolp.add_parser('conclude', help='Show the execution statistics')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--puzzle_name', help='Only show one specific testing puzzle', default='')
    p.add_argument('--type', help='Choose what kind of info to output', default='detail', choices=['detail', 'stat'])
    p.add_argument('--out', help='Output CSV file', default='1.csv')
    p.add_argument('dirs', help='Workspace directory', nargs='+')

    p = toolp.add_parser('breakdown', help='Show the per-stage runtime')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--puzzle_name', help='Only show one specific testing puzzle', default='')
    p.add_argument('--type', help='Choose what kind of info to output', default='detail', choices=['detail', 'stat'])
    p.add_argument('--out', help='Output CSV file', default='b.csv')
    p.add_argument('dirs', help='Workspace directory', nargs='+')

    p = toolp.add_parser('condor_breakdown', help='Show the solving per-stage runtime on HTCondor cluster')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--puzzle_name', help='Only show one specific testing puzzle', default='')
    p.add_argument('--out', help='Output CSV file', default='c.csv')
    p.add_argument('dirs', help='Archived workspace directory, must include condor log files',
                   nargs='+')

    p = toolp.add_parser('condor_ppbreakdown', help='Show the preprocessing per-stage runtime on HTCondor cluster')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--puzzle_name', help='Only show one specific testing puzzle', default='')
    p.add_argument('--out', help='Output CSV file', default='p.csv')
    p.add_argument('dirs', help='Archived workspace directory, must include condor log files',
                   nargs='+')

    p = toolp.add_parser('blender', help='Invoke blender to render the trajectory')
    p.add_argument('--current_trial', help='Trial that has the solution trajectory', type=int, default=None)
    p.add_argument('--puzzle_name', help='Puzzle selection. Will exit after displaying all puzzle names if not present', default='')
    p.add_argument('dir', help='Workspace directory')
    p.add_argument('outdir', help='Output directory')

    p = toolp.add_parser('dump_training_data', help='Dump the training data to output file')
    p.add_argument('--puzzle_name', help='Training Puzzle selection. The name of the default puzzle is "train".', default='')
    p.add_argument('--geo_type', help='Type of the geometry (env/rob)', default='rob')
    p.add_argument('--nn_profile', help='NN profile', default='256hg+normal')
    p.add_argument('dir', help='Workspace directory')
    p.add_argument('out', help='Output directory')

def run(args):
    function_dict[args.tool_name](args)
