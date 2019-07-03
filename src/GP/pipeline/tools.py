#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isdir, isfile
import copy
import pathlib
import csv
import numpy as np

from . import util
from . import matio
from . import condor
from . import parse_ompl
from . import atlas
from . import partt

def read_roots(args):
    uw = util.create_unit_world(args.puzzle_fn)
    ompl_q = matio.load(args.roots, key=args.roots_key)
    print("OMPL_Q {}".format(ompl_q.shape))
    unit_q = uw.translate_ompl_to_unit(ompl_q)
    matio.savetxt(args.out, unit_q)

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
            for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
                if args.puzzle_name and args.puzzle_name != puzzle_name:
                    continue
                key_fn = ws.keyconf_prediction_file(puzzle_name, for_read=False)
                if not isfile(key_fn):
                    util.log("[viskey] Could not find {}".format(key_fn))
                    continue
                yield puzzle_fn, puzzle_name, key_fn
        return
    for puzzle_fn, puzzle_name, key_fn in puzgen(args):
        keys = matio.load(key_fn)['KEYQ_OMPL']
        uw = util.create_unit_world(puzzle_fn)
        if args.range:
            keys = keys[util.rangestring_to_list(args.range)]
        ukeys = uw.translate_ompl_to_unit(keys)
        matio.savetxt('viskey.tmp.txt', ukeys)

        cfg, _ = parse_ompl.parse_simple(puzzle_fn)
        util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'viskey.tmp.txt', '0.5'])

def visimp(args):
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
        matio.savetxt('visimp.tmp.txt', np.concatenate((unit_imp_1, unit_imp_2), axis=0))
        matio.savetxt('visimp.vanilla.txt', np.concatenate((pts[:, 0:3], pts[:,3:6]), axis=0))
        util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'visimp.tmp.txt', '0.5'])


def visnotch(args):
    # FIXME: this function is mostly copied from visimp
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
            pts = kpp.probe_notch_points(args.pairs)
            if pts.shape[0] > 0:
                break
            util.log("Found 0 key points, try again")
        uw = util.create_unit_world(wag.puzzle_fn)
        util.log("Found {} key points".format(pts.shape[0]))
        unit_imp_1 = uw.translate_vanilla_pts_to_unit(uw.GEO_ROB, pts[:, 0:3])
        unit_imp_2 = uw.translate_vanilla_pts_to_unit(uw.GEO_ROB, pts[:, 3:6])
        matio.savetxt('visimp.tmp.txt', np.concatenate((unit_imp_1, unit_imp_2), axis=0))
        matio.savetxt('visimp.vanilla.txt', np.concatenate((pts[:, 0:3], pts[:,3:6]), axis=0))
        util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'visimp.tmp.txt', '0.5'])


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
    if key in dic:
        dic[key].append(v)
    else:
        dic[key] = [v]

def _print_stat(puzzle_name, stat_dic, f):
    writer = csv.writer(f)
    for i in range(len(stat_dic['trial_id'])):
        writer.writerow([puzzle_name,
                         stat_dic['trial_id'][i],
                         stat_dic['puzzle_method'][i],
                         '{}/{}'.format(stat_dic['puzzle_kps_env'][i],
                                        stat_dic['puzzle_kps_rob'][i]),
                         stat_dic['puzzle_rot'][i],
                         stat_dic['puzzle_roots'][i],
                         stat_dic['puzzle_pds'][i],
                         stat_dic['puzzle_success'][i]])
    writer.writerow([puzzle_name, stat_dic['trial_range'],
                     stat_dic['puzzle_method'][0] if all(elem == stat_dic['puzzle_method'][0] for elem in stat_dic['puzzle_method']) else '*MIXED*',
                     'mean: {}/{}'.format(np.mean(stat_dic['puzzle_kps_env']),
                                          np.mean(stat_dic['puzzle_kps_rob'])),
                     'mean: {}'.format(np.mean(stat_dic['puzzle_rot'])),
                     'mean: {}'.format(np.mean(stat_dic['puzzle_roots'])),
                     'mean: {}'.format(np.mean(stat_dic['puzzle_pds'])),
                     '{}/{}'.format(np.sum([1 if e == 'Y' else 0 for e in stat_dic['puzzle_success']]), len(stat_dic['puzzle_success']))
                    ])

def conclude(args):
    f = open(args.out, 'w')
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
                    break
                kp_env_fn = ws.keypoint_prediction_file(puzzle_name, 'env')
                kp_rob_fn = ws.keypoint_prediction_file(puzzle_name, 'rob')
                if os.path.exists(kp_rob_fn) and os.path.exists(kp_env_fn):
                    puzzle_method = 'GK'
                    puzzle_rot = ws.config.getint('GeometriK', 'KeyConfigRotations')
                    d_env = matio.load(kp_env_fn)
                    d_rob = matio.load(kp_rob_fn)
                    puzzle_kps_env = util.access_keypoints(d_env).shape[0]
                    puzzle_kps_rob = util.access_keypoints(d_rob).shape[0]
                else:
                    puzzle_method = 'NN'
                    puzzle_rot = ws.config.getint('Prediction', 'NumberOfRotations')
                    puzzle_kps = 'N/A'
                kq_fn = ws.keyconf_prediction_file(puzzle_name)
                puzzle_roots = matio.load(kq_fn)['KEYQ_OMPL'].shape[0]
                puzzle_pds = matio.load(pds_fn)['Q'].shape[0]
                sol_fn = ws.solution_file(puzzle_name, type_name='unit')
                if os.path.exists(sol_fn):
                    puzzle_success = 'Y'
                else:
                    puzzle_success = 'N'
                _dic_add(stat_dic, 'trial_id', trial)
                _dic_add(stat_dic, 'puzzle_method', puzzle_method)
                _dic_add(stat_dic, 'puzzle_kps_env', puzzle_kps_env)
                _dic_add(stat_dic, 'puzzle_kps_rob', puzzle_kps_rob)
                _dic_add(stat_dic, 'puzzle_rot', puzzle_rot)
                _dic_add(stat_dic, 'puzzle_roots', puzzle_roots)
                _dic_add(stat_dic, 'puzzle_pds', puzzle_pds)
                _dic_add(stat_dic, 'puzzle_success', puzzle_success)
            stat_dic['trial_range'] = args.trial_range
            _print_stat(puzzle_name, stat_dic, f=f)
    f.close()

function_dict = {
        'read_roots' : read_roots,
        'visenvgt' : visenvgt,
        'visrobgt' : visrobgt,
        'vistraj' : vistraj,
        'plotstat' : plotstat,
        'visclearance' : visclearance,
        'visnnpred' : visnnpred,
        'visnnsample' : visnnsample,
        'viskey' : viskey,
        'visimp' : visimp,
        'visnotch' : visnotch,
        'vistouchv' : vistouchv,
        'vistouchdisp' : vistouchdisp,
        'animate' : animate,
        'conclude' : conclude,
}

def setup_parser(subparsers):
    sp = subparsers.add_parser('tools', help='Various Tools.')
    toolp = sp.add_subparsers(dest='tool_name', help='Name of Tool')
    p = toolp.add_parser('read_roots', help='Dump roots of the forest to text file')
    p.add_argument('--roots_key', help='NPZ file of roots', default='KEYQ_OMPL')
    p.add_argument('puzzle_fn', help='OMPL config')
    p.add_argument('roots', help='NPZ file of roots')
    p.add_argument('out', help='output txt file')
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
    p.add_argument('--current_trial', help='Trial to predict the keyconf', type=int, default=None)
    p.add_argument('--range', help='Range of key confs, e.g. 1,2,3,4-7,11', default='')
    p.add_argument('--puzzle_name', help='Only show one specific puzzle', default='')
    p.add_argument('dir', help='Workspace directory')
    p = toolp.add_parser('visimp', help='Visualize "Important Points" from geometric hueristics')
    p.add_argument('--pairs', help='How many pairs of points to generate', type=int, default=12)
    p.add_argument('dir', help='Workspace directory')
    p = toolp.add_parser('visnotch', help='Visualize Notch from geometric hueristics')
    p.add_argument('--pairs', help='How many pairs of points to generate', type=int, default=12)
    p.add_argument('dir', help='Workspace directory')
    p = toolp.add_parser('vistouchv', help='Visualize the touch configurations in clearance estimation')
    p.add_argument('--key_id', help='Key ID in KeyCan.npz. Top K can be found at KEY.npz:_TOP_K', type=int, required=True)
    p.add_argument('dir', help='Workspace directory')
    p = toolp.add_parser('vistouchdisp', help='Visualize the displacement of touch configurations from clearance estimation')
    p.add_argument('--key_id', help='Key ID in KeyCan.npz. Top K can be found at KEY.npz:_TOP_K', type=int, required=True)
    p.add_argument('dir', help='Workspace directory')
    p = toolp.add_parser('animate', help='Show the animation of solution with vispath')
    p.add_argument('--current_trial', help='Trial to predict the keyconf', type=int, default=None)
    p.add_argument('--puzzle_name', help='Only show one specific puzzle', default='')
    p.add_argument('dir', help='Workspace directory')
    p = toolp.add_parser('conclude', help='Show the execution statistics')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--puzzle_name', help='Only show one specific testing puzzle', default='')
    p.add_argument('--out', help='Output CSV file', default='1.csv')
    p.add_argument('dirs', help='Workspace directory', nargs='+')

def run(args):
    function_dict[args.tool_name](args)
