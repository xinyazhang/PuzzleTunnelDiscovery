#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isdir, isfile
import subprocess
import pathlib
import numpy as np
import copy
import multiprocessing
from imageio import imwrite as imsave
from imageio import imread
from progressbar import progressbar, ProgressBar
import shutil

from . import util
from . import disjoint_set
from . import choice_formatter
try:
    from . import se3solver
except ImportError as e:
    util.warn(str(e))
    util.warn("[WARNING] CANNOT IMPORT se3solver. Some function will be disabled and the pipeline is broken")
from . import partt
from . import condor
from . import matio
from . import atlas
from . import texture_format
from . import parse_ompl

ALGORITHM_VERSION_PHASE2_WITH_BLOOMING_TREE = 5

#
# Functions to process workspaces locally
#

def _trial_id(ws, trial):
    return str(trial)
    '''
    max_trial = ws.config.getint('Solver', 'Trials', fallback=1)
    # util.log('[_trial_id] trial {} max_trial {}'.format(trial, max_trial))
    return util.padded(trial, max(trial, max_trial))
    '''

def _puzzle_pds(ws, puzzle_name, trial):
    pds_dir = ws.local_ws(util.SOLVER_SCRATCH,
                          puzzle_name,
                          util.PDS_SUBDIR)
    os.makedirs(pds_dir, exist_ok=True)
    fn ='{}.npz'.format(_trial_id(ws, trial))
    return join(pds_dir, fn)

def _rel_bloom_scratch(ws, puzzle_fn, trial):
    return join(util.SOLVER_SCRATCH, puzzle_fn, util.PDS_SUBDIR, 'bloom-'+_trial_id(ws, trial))

def _partition_screening(ws, puzzle_name, index=None):
    keyfn = ws.keyconf_prediction_file(puzzle_name)
    keys = matio.load(keyfn, key='KEYQ_OMPL')
    nkey = keys.shape[0] - util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS
    task_indices = np.tril_indices(nkey)
    task_shape = task_indices[0].shape
    total_chunks = partt.guess_chunk_number(task_shape,
            ws.config.getint('DEFAULT', 'CondorQuota') * 4,
            ws.config.getint('TrainingKeyConf', 'ClearanceTaskGranularity'))
    if index is None:
        return keys, total_chunks
    chunk = partt.get_task_chunk(task_shape, total_chunks, index)
    chunk = np.array(chunk).reshape(-1)
    # util.log('chunk shape {}'.format(chunk.shape))
    return (keys,
            task_indices[0][chunk] + util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS,
            task_indices[1][chunk] + util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS)

def screen_keyconf(args, ws):
    screen_str = 'screen-{}'.format(ws.current_trial)
    if args.task_id is not None:
        # actual worker
        for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
            rel_scratch_dir = os.path.join(util.SOLVER_SCRATCH, puzzle_name, screen_str)
            scratch_dir = ws.local_ws(rel_scratch_dir)

            keys, from_indices, to_indices = _partition_screening(ws,
                                                                  puzzle_name,
                                                                  index=args.task_id)
            # util.log('keys shape {}'.format(keys.shape))
            # util.log('from_indices shape {}'.format(from_indices.shape))
            uw = util.create_unit_world(puzzle_fn)
            # util.log('from {}'.format(keys[from_indices].shape))
            # util.log('from_indices[:16] {}'.format(from_indices[:16]))
            # util.log('to_indices[:16] {}'.format(to_indices[:16]))
            visb_pairs = uw.calculate_visibility_pair(keys[from_indices], False,
                                                      keys[to_indices], False,
                                                      uw.recommended_cres,
                                                      enable_mt=False)
            visb_vec = visb_pairs.reshape((-1))
            visibile_indices = visb_vec.nonzero()[0]
            outfn = join(scratch_dir, 'edge_batch-{}.npz'.format(args.task_id))
            np.savez_compressed(outfn, EDGE_FROM=from_indices[visibile_indices], EDGE_TO=to_indices[visibile_indices])
        return # worker never wait
    if not args.only_wait:
        # submit condor job
        for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
            rel_scratch_dir = os.path.join(util.SOLVER_SCRATCH, puzzle_name, screen_str)
            scratch_dir = ws.local_ws(rel_scratch_dir)
            _, total_chunks = _partition_screening(ws, puzzle_name, index=None)
            condor_job_args = [ws.condor_local_exec('facade.py'),
                               'solve',
                               '--stage', 'screen_keyconf',
                               '--current_trial', str(ws.current_trial),
                               '--puzzle_name', puzzle_name,
                               '--task_id', '$(Process)',
                               ws.local_ws()]
            condor.local_submit(ws,
                                util.PYTHON,
                                iodir_rel=rel_scratch_dir,
                                arguments=condor_job_args,
                                instances=total_chunks,
                                wait=False)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        rel_scratch_dir = os.path.join(util.SOLVER_SCRATCH, puzzle_name, screen_str)
        scratch_dir = ws.local_ws(rel_scratch_dir)
        if not args.no_wait: # wait the condor_job
            condor.local_wait(scratch_dir)

        keyfn = ws.keyconf_prediction_file(puzzle_name)
        keys = matio.load(keyfn, key='KEYQ_OMPL')
        nkey = keys.shape[0]
        util.log('[screen_keyconf][{}] nkey (unscreened) {}'.format(puzzle_name, nkey))
        vert_ids = [i for i in range(util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS, nkey)]
        djs = disjoint_set.DisjointSet(vert_ids)
        fn_list = pathlib.Path(scratch_dir).glob("edge_batch-*.npz")
        util.log("[screen_keyconf][{}] Creating disjoint set".format(puzzle_name))
        for fn in progressbar(fn_list):
            d = matio.load(fn)
            for r,c in zip(d['EDGE_FROM'], d['EDGE_TO']):
                #if str(fn) == '/u/zxy/scratch/auto-mkobs3d/bin/gkws/duet-g1/solver_scratch/duet-g1/screen-80/edge_batch-120.npz':
                # util.log("{} {} {} {}".format(r, c, djs.find(r), djs.find(c)))
                djs.union(r,c)

        cluster = djs.get_cluster()
        screened_index = []
        uw = util.create_unit_world(puzzle_fn)
        unit_keys = uw.translate_ompl_to_unit(keys)
        util.log("[screen_keyconf][{}] Screening roots".format(puzzle_name))
        for k in progressbar(cluster):
            nondis = []
            for member in cluster[k]:
                if not uw.is_disentangled(unit_keys[member]):
                    nondis.append(member)
                    break
            # TODO: clustering?
            screened_index += nondis # No effect if nondis == []
        screened_index = list(range(util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS)) + screened_index
        screened = keys[screened_index]
        util.log("[screen_keyconf][{}] Screened {} roots into {}".format(puzzle_name,
                  keys.shape, screened.shape))
        outfn = ws.screened_keyconf_prediction_file(puzzle_name)
        util.log("[screen_keyconf] Save screened roots {} to {}".format(screened.shape, outfn))
        np.savez_compressed(outfn, KEYQ_OMPL=screened)

'''
least_visible_keyconf:
    this stage would wait for the finish of keyconf.calculate_keyconf_clearance,
    and pick up K key confs with lowest clearance.
'''
def least_visible_keyconf(args, ws):
    # wait for all key confs
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        rel_scratch_dir = join(util.SOLVER_SCRATCH, puzzle_name, util.KEYCONF_CLEARANCE_DIR, str(ws.current_trial))
        condor.local_wait(ws.local_ws(rel_scratch_dir))

    K = ws.config.getint('Prediction', 'SurfacePairsToSample')
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        rel_scratch_dir = join(util.SOLVER_SCRATCH, puzzle_name, util.KEYCONF_CLEARANCE_DIR, str(ws.current_trial))
        oskey_fn = ws.oversampled_keyconf_prediction_file(puzzle_name)
        oskey = matio.load(oskey_fn)['KEYQ_OMPL']
        # TODO: Pickup top key confs
        '''
        '''
        osc_files = util.lsv(ws.local_ws(rel_scratch_dir), 'clearance_batch-', '.npz')
        osc_arrays = [matio.load(fn)['DISTANCE_BATCH'] for fn in osc_files]
        osc = util.safe_concatente(osc_arrays)
        assert osc.shape[0] == oskey.shape[0], 'osc shape {} != oskey shape {}'.format(osc.shape, oskey.shape)
        mean = np.mean(osc, axis=1)
        # remove the initial root and goal root
        mean = mean[util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS:]
        top_k = mean.argsort()[:K]
        # Get the original indices
        top_k += util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS
        top_oskey = oskey[top_k,:]
        gk_key = np.load(ws.keyconf_file_from_fmt(puzzle_name, util.GEOMETRIK_KEY_PREDICTION_FMT))['KEYQ_OMPL']
        # also remove the initial and goal roots
        gk_key = gk_key[util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS:]
        uskey_fn = ws.keyconf_prediction_file(puzzle_name)
        ig_key = oskey[:util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS]
        comb_key = util.safe_concatente([ig_key, top_oskey, gk_key])
        np.savez(uskey_fn, KEYQ_OMPL=comb_key)
        util.ack('[least_visible_keyconf] save combined key to {}, shape {}'.format(uskey_fn, comb_key.shape))

class TmpDriverArgs(object):
    pass

def _sample_pds_old(args, ws):
    if 'pipeline.se3solver' not in sys.modules:
        raise RuntimeError("se3solver is not loaded")
    max_trial = ws.config.getint('Solver', 'Trials', fallback=1)
    nsamples = ws.config.getint('Solver', 'PDSSize')
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        util.log('[sample_pds] sampling puzzle {}'.format(puzzle_name))
        driver_args = TmpDriverArgs()
        driver_args.puzzle = puzzle_fn
        driver_args.planner_id = se3solver.PLANNER_PRM
        driver_args.sampler_id = 0
        driver = se3solver.create_driver(driver_args)
        uw = util.create_unit_world(puzzle_fn)
        for i in range(max_trial):
            # util.log('[sample_pds] trial id {}'.format(_trial_id(ws, 0)))
            Q = driver.presample(nsamples)
            uQ = uw.translate_ompl_to_unit(Q)
            n = Q.shape[0]
            QF = np.zeros((n, 1), dtype=np.uint32)
            for j, uq in enumerate(uQ):
                if uw.is_disentangled(uq):
                    QF[j] = se3solver.PDS_FLAG_TERMINATE
            fn = _puzzle_pds(ws, puzzle_name, i)
            np.savez(fn, Q=Q, QF=QF)
            util.log('[sample_pds] samples stored at {}'.format(fn))

# Bloom from roots
def sample_pds(args, ws):
    if not args.only_wait:
        for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
            _, config = parse_ompl.parse_simple(puzzle_fn)
            rel_bloom = _rel_bloom_scratch(ws, puzzle_name, ws.current_trial)
            util.log('[sample_pds]  rel_bloom {}'.format(rel_bloom))
            scratch_dir = ws.local_ws(rel_bloom)
            os.makedirs(scratch_dir, exist_ok=True)
            key_fn = ws.screened_keyconf_prediction_file(puzzle_name)
            condor_job_args = ['se3solver.py',
                    'solve',
                    '--cdres', config.getfloat('problem', 'collision_resolution', fallback=0.0001),
                    '--replace_istate',
                    'file={},key=KEYQ_OMPL,offset=$$([$(Process)]),size=1,out={}'.format(key_fn, scratch_dir),
                    '--bloom_out',
                    join(scratch_dir, 'bloom-from_$(Process).npz'),
                    puzzle_fn,
                    util.RDT_FOREST_ALGORITHM_ID,
                    0.25,
                    '--bloom_limit',
                    ws.config.getint('Solver', 'PDSBloom')
                              ]
            keys = matio.load(key_fn)
            condor.local_submit(ws,
                                util.PYTHON,
                                iodir_rel=rel_bloom,
                                arguments=condor_job_args,
                                instances=keys['KEYQ_OMPL'].shape[0],
                                wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
    if args.no_wait:
        return

def assemble_pds(args, ws):
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        rel_bloom = _rel_bloom_scratch(ws, puzzle_name, ws.current_trial)
        scratch_dir = ws.local_ws(rel_bloom)
        condor.local_wait(scratch_dir)
        pds_fn = _puzzle_pds(ws, puzzle_name, ws.current_trial)
        Q_list = []
        QE_list = []
        tree_base_list = []
        edge_base_list = []
        fn_list = sorted(pathlib.Path(scratch_dir).glob("bloom-from_*.npz"))
        tree_base = 0
        edge_base = 0
        for fn in progressbar(fn_list):
            d = matio.load(fn)
            s = d['BLOOM'].shape
            if s[0] == 0:
                continue
            assert s[1] == 7, "{}'s shape is {}".format(fn, s)
            bloom = d['BLOOM']
            Q_list.append(bloom)
            if 'BLOOM_EDGE' in d and QE_list is not None:
                edges = np.transpose(d['BLOOM_EDGE']) + tree_base
                QE_list.append(edges)
                edge_base_list.append(edge_base)
                edge_base += int(edges.shape[0])
            else:
                QE_list = None # Disable QE
            tree_base_list.append(tree_base)
            tree_base += int(bloom.shape[0])
        Q = np.concatenate(Q_list, axis=0)
        uw = util.create_unit_world(puzzle_fn)
        uQ = uw.translate_ompl_to_unit(Q)
        QF = np.zeros((Q.shape[0], 1), dtype=np.uint32)
        for j, uq in enumerate(uQ):
            if uw.is_disentangled(uq):
                QF[j] = se3solver.PDS_FLAG_TERMINATE
        if QE_list:
            QE = np.concatenate(QE_list, axis=0)
            assert QE.shape[0] == Q.shape[0] - len(tree_base_list), 'San check failed. Broken tree edges'
            np.savez_compressed(pds_fn, Q=Q, QF=QF, QB=tree_base_list, QE=QE, QEB=edge_base_list)
        else:
            np.savez_compressed(pds_fn, Q=Q, QF=QF, QB=tree_base_list)
        util.log('[sample_pds] samples stored at {}'.format(pds_fn))

def forest_rdt(args, ws):
    if args.algorithm_version == ALGORITHM_VERSION_PHASE2_WITH_BLOOMING_TREE:
        algoprefix = 'withbt-'
    else:
        algoprefix = ''
    trial_str = algoprefix + 'trial-{}'.format(_trial_id(ws, ws.current_trial))
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        # if --puzzle_name presents, skip unmatched puzzles
        if args.puzzle_name and args.puzzle_name != puzzle_name:
            continue
        rel_scratch_dir = os.path.join(util.SOLVER_SCRATCH, puzzle_name, trial_str)
        _, config = parse_ompl.parse_simple(puzzle_fn)
        scratch_dir = ws.local_ws(rel_scratch_dir)
        if args.only_wait:
            condor.local_wait(scratch_dir)
            continue
        # se3solver.py solve
        # /u/zxy/applique/puzzle-geometry/puzzles/claw/claw.cfg 15 1.00
        # --samset mkobs-claw/pds/1m.0.npz --replace_istate
        # file=mkobs-claw/key.ompl.withig.npz,offset=$$([$(Process)*1]),size=1,out=mkobs-claw/trial-1/pdsrdt/
        #key_fn = ws.local_ws(util.TESTING_DIR, puzzle_name, util.KEY_PREDICTION)

        # Note: no need to probe the key configuration, which has been done in sample_pds and
        #       ws.screened_keyconf_prediction_file should always point to a valid keyconf
        key_fn = ws.screened_keyconf_prediction_file(puzzle_name)
        condor_job_args = ['se3solver.py',
                'solve',
                '--cdres', config.getfloat('problem', 'collision_resolution', fallback=0.0001),
                '--samset', _puzzle_pds(ws, puzzle_name, ws.current_trial),
                '--replace_istate',
                'file={},key=KEYQ_OMPL,offset=$$([$(Process)]),size=1,out={}'.format(key_fn, scratch_dir)]
        if args.algorithm_version == ALGORITHM_VERSION_PHASE2_WITH_BLOOMING_TREE:
            condor_job_args += ['--use_blooming_tree']
        condor_job_args += [puzzle_fn,
                util.RDT_FOREST_ALGORITHM_ID,
                1.0]
        keys = matio.load(key_fn)
        condor.local_submit(ws,
                            util.PYTHON,
                            iodir_rel=rel_scratch_dir,
                            arguments=condor_job_args,
                            instances=keys['KEYQ_OMPL'].shape[0],
                            wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
    if args.no_wait:
        return
    if not args.only_wait:
        only_wait_args = copy.deepcopy(args)
        only_wait_args.only_wait = True
        forest_rdt(only_wait_args, ws)

def forest_edges(args, ws):
#pdsrdt-g9ae-1-edges.hdf5:
#    ./pds_edge.py --pdsflags dual/pdsrdt-g9ae-1/pds-4m.0.npz --out pdsrdt-g9ae-1-edges.hdf5 `ls -v dual/pdsrdt-g9ae-1/pdsrdt/ssc-*.mat`
    if args.algorithm_version == ALGORITHM_VERSION_PHASE2_WITH_BLOOMING_TREE:
        algoprefix = 'withbt-'
    else:
        algoprefix = ''
    trial_str = algoprefix + 'trial-{}'.format(_trial_id(ws, ws.current_trial))
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        if args.puzzle_name and args.puzzle_name != puzzle_name:
            continue
        rel_scratch_dir = os.path.join(util.SOLVER_SCRATCH, puzzle_name, trial_str)
        shell_script = './pds_edge.py --pdsflags '
        shell_script += _puzzle_pds(ws, puzzle_name, ws.current_trial)
        shell_script += ' --out {}'.format(ws.local_ws(rel_scratch_dir, algoprefix + 'edges.hdf5'))
        shell_script += ' `ls -v {}/ssc-*.mat`'.format(ws.local_ws(rel_scratch_dir))
        util.shell(['bash', '-c', shell_script])

def connect_forest(args, ws):
# pdsrdt-g9ae-1-path-1.txt: pdsrdt-g9ae-1-edges-1.hdf5
#   ./forest_dijkstra.py --indir dual/pdsrdt-g9ae-1/pdsrdt-1/ \
#               --forest_edge pdsrdt-g9ae-1-edges-1.hdf5 \
#               --rootf dual/pdsrdt-g9ae-1/postscreen-1_withig.npz \
#               --pdsf dual/pdsrdt-g9ae-1/pds-4m.0.npz \
#               --out pdsrdt-g9ae-1-path-1.txt
#
    if args.algorithm_version == ALGORITHM_VERSION_PHASE2_WITH_BLOOMING_TREE:
        algoprefix = 'withbt-'
    else:
        algoprefix = ''
    trial_str = algoprefix + 'trial-{}'.format(_trial_id(ws, ws.current_trial))
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        if args.puzzle_name and args.puzzle_name != puzzle_name:
            continue
        # key_fn = ws.local_ws(util.TESTING_DIR, puzzle_name, util.KEY_PREDICTION)
        key_fn = ws.screened_keyconf_prediction_file(puzzle_name)
        rel_scratch_dir = join(util.SOLVER_SCRATCH, puzzle_name, trial_str)
        rel_edges = join(rel_scratch_dir, algoprefix + 'edges.hdf5')
        path_out = ws.local_ws(rel_scratch_dir, algoprefix + 'path.txt')
        shell_script = './forest_dijkstra.py'
        shell_script += ' --indir '
        shell_script += ws.local_ws(rel_scratch_dir)
        shell_script += ' --forest_edge '
        shell_script += ws.local_ws(rel_edges)
        shell_script += ' --rootf '
        shell_script += key_fn
        shell_script += ' --pdsf '
        shell_script += _puzzle_pds(ws, puzzle_name, ws.current_trial)
        shell_script += ' --out {}'.format(path_out)
        ret = util.shell(['bash', '-c', shell_script])
        if ret != 0:
            util.fatal("[solve] FALIED TO SOLVE PUZZLE {}".format(puzzle_name))
            continue
        util.ack("Saving OMPL solution of {} to {}".format(puzzle_name, path_out))

        ompl_q = matio.load(path_out)
        uw = util.create_unit_world(puzzle_fn)
        unit_q = uw.translate_ompl_to_unit(ompl_q)
        sol_out = ws.solution_file(puzzle_name, type_name='unit')
        matio.savetxt(sol_out, unit_q)
        util.ack("Saving UNIT solution of {} to {}".format(puzzle_name, sol_out))

'''
knn_forest:
    Connect forest with prm like algorithm
'''
def knn_forest(args, ws):
    ALGO_VERSION = 0 if args.algorithm_version is None else args.algorithm_version
    trial_list = util.rangestring_to_list(args.current_trial)
    trial_str = 'knn_forest-{}-algver_{}'.format(args.current_trial, ALGO_VERSION)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        rel_scratch_dir = os.path.join(util.BASELINE_SCRATCH, puzzle_name, trial_str)
        _, config = parse_ompl.parse_simple(puzzle_fn)
        scratch_dir = ws.local_ws(rel_scratch_dir)
        if args.only_wait:
            condor.local_wait(scratch_dir)
            continue
        if args.task_id is None:
            condor_job_args = ['./facade.py',
                    'solve',
                    '--stage', 'knn_forest',
                    '--current_trial', args.current_trial,
                    '--puzzle_name', puzzle_name,
                    '--algorithm_version', ALGO_VERSION,
                    '--task_id', '$(Process)',
                    ws.local_ws()]
            condor.local_submit(ws,
                                util.PYTHON,
                                iodir_rel=rel_scratch_dir,
                                arguments=condor_job_args,
                                instances=len(trial_list),
                                wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
        else:
            solver_args = TmpDriverArgs()
            solver_args.puzzle = puzzle_fn
            ws.current_trial = trial_list[args.task_id]
            rel_bloom = _rel_bloom_scratch(ws, puzzle_name, ws.current_trial)
            solver_args.bloom_dir = ws.local_ws(rel_bloom)
            solver_args.out = os.path.join(scratch_dir, 'edges-{}.npz'.format(ws.current_trial))
            solver_args.knn = 8 # default
            solver_args.algo_version = ALGO_VERSION # algorithm version
            se3solver.merge_blooming_forest(solver_args)
    if args.task_id is not None:
        return
    if args.no_wait:
        return
    if not args.only_wait:
        only_wait_args = copy.deepcopy(args)
        only_wait_args.only_wait = True
        forest_rdt(only_wait_args, ws)

function_dict = {
        'least_visible_keyconf' : least_visible_keyconf,
        'screen_keyconf' : screen_keyconf,
        'sample_pds' : sample_pds,
        'assemble_pds' : assemble_pds,
        'forest_rdt' : forest_rdt,
        'forest_edges' : forest_edges,
        'connect_forest' : connect_forest,
        'knn_forest' : knn_forest,
}

def setup_parser(subparsers):
    p = subparsers.add_parser('solve', help='Final step to solve the puzzle',
                              formatter_class=choice_formatter.Formatter)
    p.add_argument('--stage',
                   choices=list(function_dict.keys()),
                   help='R|Possible stages:\n'+'\n'.join(list(function_dict.keys())),
                   default='',
                   metavar='')
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--no_wait', action='store_true')
    p.add_argument('--task_id', help='Feed $(Process) from HTCondor', type=int, default=None)
    p.add_argument('--current_trial', help='Trial to solve the puzzle', type=str, default='0')
    p.add_argument('--puzzle_name', help='Pick a single puzzle to solve (default to all)', default='')
    p.add_argument('--algorithm_version', help='Algorithm version (varying among stages', type=int, default=None)
    p.add_argument('dir', help='Workspace directory')

def run(args):
    if args.stage in function_dict:
        ws = util.Workspace(args.dir)
        if args.stage == 'knn_forest': # knn_forest takes current_trial list instead of single trial
            function_dict[args.stage](args, ws)
        else:
            for current_trial in util.rangestring_to_list(args.current_trial):
                ws.current_trial = current_trial
                function_dict[args.stage](args, ws)
    else:
        print("Unknown solve pipeline stage {}".format(args.stage))

#
# Automatic functions start here
#
def _remote_command(ws, cmd, auto_retry=True, alter_host='', extra_args=''):
    if not alter_host:
        alter_host = ws.condor_host
    ws.remote_command(alter_host,
                      ws.condor_exec(),
                      ws.condor_ws(),
                      'solve', cmd, auto_retry=auto_retry,
                      with_trial=True,
                      extra_args=extra_args)

def _remote_command_distributed(ws, cmd):
    for host,_,puzzle_name in ws.condor_host_vs_test_puzzle_generator():
        _remote_command(ws, cmd,
                        alter_host=host,
                        extra_args='--puzzle_name {} --no_wait'.format(puzzle_name))
    for host,_,puzzle_name in ws.condor_host_vs_test_puzzle_generator():
        _remote_command(ws, cmd,
                        alter_host=host,
                        extra_args='--puzzle_name {} --only_wait'.format(puzzle_name))

def _remote_command_auto(ws, cmd):
    if ws.condor_extra_hosts:
        _remote_command_distributed(ws, cmd)
    else:
        _remote_command(ws, cmd)

def remote_screen_keyconf(ws):
    _remote_command_auto(ws, 'screen_keyconf')

def remote_least_visible_keyconf(ws):
    _remote_command_auto(ws, 'least_visible_keyconf')

def remote_sample_pds(ws):
    _remote_command_auto(ws, 'sample_pds')

def remote_assemble_pds(ws):
    _remote_command_auto(ws, 'assemble_pds')

def remote_forest_rdt(ws):
    _remote_command_auto(ws, 'forest_rdt')

def remote_forest_edges(ws):
    for _,_,puzzle_name in ws.condor_host_vs_test_puzzle_generator():
        ws.timekeeper_start('forest_edges', puzzle_name)
        _remote_command(ws, 'forest_edges',
                        extra_args='--puzzle_name {}'.format(puzzle_name))
        ws.timekeeper_finish('forest_edges', puzzle_name)

def remote_connect_forest(ws):
    for _,_,puzzle_name in ws.condor_host_vs_test_puzzle_generator():
        ws.timekeeper_start('connect_forest', puzzle_name)
        _remote_command(ws, 'connect_forest',
                        extra_args='--puzzle_name {}'.format(puzzle_name))
        ws.timekeeper_finish('connect_forest', puzzle_name)

def remote_forest_rdt_withbt(ws):
    _remote_command_auto(ws, 'forest_rdt', extra_args='--algorithm_version {}'.format(ALGORITHM_VERSION_PHASE2_WITH_BLOOMING_TREE))

def remote_forest_edges_withbt(ws):
    for _,_,puzzle_name in ws.condor_host_vs_test_puzzle_generator():
        ws.timekeeper_start('forest_edges', puzzle_name)
        _remote_command(ws, 'forest_edges',
                        extra_args='--puzzle_name {} --algorithm_version {}'.format(puzzle_name, ALGORITHM_VERSION_PHASE2_WITH_BLOOMING_TREE))
        ws.timekeeper_finish('forest_edges', puzzle_name)

def remote_connect_forest_withbt(ws):
    for _,_,puzzle_name in ws.condor_host_vs_test_puzzle_generator():
        ws.timekeeper_start('connect_forest', puzzle_name)
        _remote_command(ws, 'connect_forest',
                        extra_args='--puzzle_name {} --algorithm_version {}'.format(puzzle_name, ALGORITHM_VERSION_PHASE2_WITH_BLOOMING_TREE))
        ws.timekeeper_finish('connect_forest', puzzle_name)


def collect_stages(variant=0):
    assert variant in [0,4,5], f'Solve Pipeline Variant {variant} has not been implemented'
    if variant in [0]:
        ret = [
                ('screen_keyconf', remote_screen_keyconf),
                ('sample_pds', remote_sample_pds),
                ('assemble_pds', remote_assemble_pds),
                ('forest_rdt', remote_forest_rdt),
                ('forest_edges', remote_forest_edges),
                ('connect_forest', remote_connect_forest),
              ]
    elif variant in [4]:
        ret = [
                ('least_visible_keyconf', remote_least_visible_keyconf),
                ('screen_keyconf', remote_screen_keyconf),
                ('sample_pds', remote_sample_pds),
                ('assemble_pds', remote_assemble_pds),
                ('forest_rdt', remote_forest_rdt),
                ('forest_edges', remote_forest_edges),
                ('connect_forest', remote_connect_forest),
              ]
    elif variant in [5]:
        ret = [
                ('assemble_pds', remote_assemble_pds),
                ('forest_rdt_withbt', remote_forest_rdt_withbt),
                ('forest_edges_withbt', remote_forest_edges_withbt),
                ('connect_forest_withbt', remote_connect_forest_withbt),
              ]
    return ret

def autorun(args):
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    pdesc = collect_stages()
    for _,func in pdesc:
        func(ws)

