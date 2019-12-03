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
from .solve import (
        setup_parser as original_setup_parser
)
from .file_locations import RAW_KEY_PRED_SCHEMES, KEY_PRED_SCHEMES, SCHEME_TO_FMT, FileLocations

class TmpDriverArgs(object):
    pass

def create_driver(puzzle_fn):
    driver_args = TmpDriverArgs()
    driver_args.puzzle = puzzle_fn
    driver_args.planner_id = se3solver.PLANNER_RDT
    driver_args.sampler_id = 0
    driver = se3solver.create_driver(driver_args)
    return driver

def remove_invalid(driver, ompl_q):
    valids = driver.validate_states(ompl_q)
    indices = valids.reshape((-1)).nonzero()[0]
    return ompl_q[indices, :]

class ScreeningPartition(object):
    def __init__(self, ws, key_fn):
        util.log(f'loading {keyfn}')
        self.keys = matio.load(keyfn)['KEYQ_OMPL']
        self.nkey = keys.shape[0] - util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS
        self.task_indices = np.tril_indices(nkey)
        self.task_shape = task_indices[0].shape
        self.total_chunks = partt.guess_chunk_number(task_shape,
                ws.config.getint('SYSTEM', 'CondorQuota') * 4,
                ws.config.getint('TrainingKeyConf', 'ClearanceTaskGranularity'))
        self.ws = ws
        self.chunk = None

    def get(self, index):
        if self.chunk is None:
            self.chunk = partt.get_task_chunk(self.task_shape, self.total_chunks, index)
            self.chunk = np.array(self.chunk).reshape(-1)
        return (self.task_indices[0][self.chunk] + util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS,
                self.task_indices[1][self.chunk] + util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS)

def _partition_screening(ws, keyfn, index=None):
    util.log(f'loading {keyfn}')
    keys = matio.load(keyfn)['KEYQ_OMPL']
    nkey = keys.shape[0] - util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS
    task_indices = np.tril_indices(nkey)
    task_shape = task_indices[0].shape
    total_chunks = partt.guess_chunk_number(task_shape,
            ws.config.getint('SYSTEM', 'CondorQuota') * 4,
            ws.config.getint('TrainingKeyConf', 'ClearanceTaskGranularity'))
    if index is None:
        return keys, total_chunks
    chunk = partt.get_task_chunk(task_shape, total_chunks, index)
    chunk = np.array(chunk).reshape(-1)
    # util.log('chunk shape {}'.format(chunk.shape))
    return (keys,
            task_indices[0][chunk] + util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS,
            task_indices[1][chunk] + util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS)

def least_visible_keyconf_fixed(args, ws):
    if args.rerun:
        util.warn('[least_visible_keyconf_fixed] --rerun is not supported')
    # wait for all key confs
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        fl = FileLocations(args, ws, puzzle_name)
        condor.local_wait(fl.clearance)

    K = ws.config.getint('Prediction', 'SurfacePairsToSample')
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        fl = FileLocations(args, ws, puzzle_name)

        oskey_fn = ws.oversampled_keyconf_prediction_file(puzzle_name)
        oskey = matio.load(oskey_fn)['KEYQ_OMPL']
        # TODO: Pickup top key confs
        '''
        '''
        osc_files, eoc = util.lsv2(fl.clearance, 'clearance_batch-', '.npz')
        util.log(f"[least_visible_keyconf_fixed][{puzzle_name}] load file from 0 to {eoc}")
        osc_arrays = [matio.load(fn)['DISTANCE_BATCH'] for fn in osc_files]
        osc = util.safe_concatente(osc_arrays)
        assert osc.shape[0] == oskey.shape[0], 'osc shape {} != oskey shape {}'.format(osc.shape, oskey.shape)
        mean = np.mean(osc ** 2, axis=1)
        # remove the initial root and goal root
        mean = mean[util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS:]
        top_k = mean.argsort()[:K]
        # Get the original indices
        top_k += util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS
        top_oskey = oskey[top_k,:]
        np.savez(fl.downsampled_key_fn, KEYQ_OMPL=top_oskey)
        util.ack(f'[least_visible_keyconf_fixed] save top {K} key to {fl.downsampled_key_fn}, shape {top_oskey.shape}')

def assemble_raw_keyconf(args, ws):
    if args.rerun:
        util.warn('[assemble_raw_keyconf] --rerun is not supported')
    if args.no_wait:
        return
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        # driver = create_driver(puzzle_fn)
        fl = FileLocations(args, ws, puzzle_name)
        keyq_dic = {}
        for scheme, fn in fl.raw_key_fn_gen:
            kq = matio.safeload(fn, 'KEYQ_OMPL')
            keyq_dic[scheme] = kq
            util.log(f'[assemble_raw_keyconf] loaded {kq.shape} from {fn} (scheme {scheme})')
        save_dic = {}
        keyq_list = []
        base = 0
        base_list = []
        for scheme in RAW_KEY_PRED_SCHEMES:
            keyq = keyq_dic[scheme]
            #keyq = remove_invalid(driver, keyq)
            nkey = keyq.shape[0]
            if nkey != 0 and base != 0: # strip off the IG states
                keyq = keyq[util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS:]
                nkey = keyq.shape[0]
            keyq_list.append(keyq)
            if base == 0:
                base_list.append(util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS if nkey > 0 else 0)
            else:
                base_list.append(base)
            base += nkey
        base_list.append(base)
        save_dic['KEYQ_OMPL'] = util.safe_concatente(keyq_list)
        save_dic['BASES_WITH_END'] = base_list
        np.savez(fl.assembled_raw_key_fn, **save_dic)
        util.ack(f'[assemble_raw_keyconf] save {save_dic["KEYQ_OMPL"].shape} to {fl.assembled_raw_key_fn}')

def screen_keyconf(args, ws):
    if args.rerun:
        for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
            fl = FileLocations(args, ws, puzzle_name)
            sp = ScreeningPartition(ws, fl.assembled_raw_key_fn)
            uw = None
            for i in range(sp.total_chunks):
                outfn = join(fl.screen, 'edge_batch-{}.npz'.format(i))
                if os.path.isfile(outfn):
                    continue
                util.log(f'<args.current_trial> [screen_keyconf][{puzzle_name}] rerun task {i}')
                if uw is None:
                    uw = util.create_unit_world(puzzle_fn)
                from_indices, to_indices = sp.get(i)
                visb_pairs = uw.calculate_visibility_pair(keys[from_indices], False,
                                                          keys[to_indices], False,
                                                          uw.recommended_cres,
                                                          enable_mt=False)
                visb_vec = visb_pairs.reshape((-1))
                visibile_indices = visb_vec.nonzero()[0]
                np.savez_compressed(outfn, EDGE_FROM=from_indices[visibile_indices], EDGE_TO=to_indices[visibile_indices])
        return
    if args.task_id is None and not args.only_wait:
        # submit condor job
        for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
            fl = FileLocations(args, ws, puzzle_name)
            _, total_chunks = _partition_screening(ws,
                                                   fl.assembled_raw_key_fn,
                                                   index=None)
            condor_job_args = [ws.condor_local_exec('facade.py'),
                               'solve2',
                               '--stage', 'screen_keyconf',
                               '--current_trial', str(ws.current_trial),
                               '--puzzle_name', puzzle_name,
                               '--scheme', 'cmb',
                               '--task_id', '$(Process)',
                               ws.local_ws()]
            condor.local_submit(ws,
                                util.PYTHON,
                                iodir_rel=fl.rel_screen,
                                arguments=condor_job_args,
                                instances=total_chunks,
                                wait=False)

    if args.task_id is not None:
        # actual worker
        for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
            fl = FileLocations(args, ws, puzzle_name)
            keys, from_indices, to_indices = _partition_screening(ws,
                                                                  fl.assembled_raw_key_fn,
                                                                  index=args.task_id)
            driver = create_driver(puzzle_fn)
            visb_pairs = driver.validate_motion_pairs(keys[from_indices], keys[to_indices])

            visb_vec = visb_pairs.reshape((-1))
            visibile_indices = visb_vec.nonzero()[0]
            outfn = join(fl.screen, 'edge_batch-{}.npz'.format(args.task_id))
            np.savez_compressed(outfn, EDGE_FROM=from_indices[visibile_indices], EDGE_TO=to_indices[visibile_indices])
        return # worker never wait

    if args.no_wait: # wait the condor_job
        return

    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        fl = FileLocations(args, ws, puzzle_name)
        condor.local_wait(fl.screen)

def assemble_roots(args, ws):
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        fl = FileLocations(args, ws, puzzle_name)
        uw = util.create_unit_world(puzzle_fn)

        keyfn = fl.assembled_raw_key_fn
        d = matio.load(keyfn)
        keys = d['KEYQ_OMPL']
        bases = d['BASES_WITH_END']
        nkey = keys.shape[0]
        '''
        Reading all edges
        '''
        util.log('[screen_keyconf][{}] nkey (unscreened) {}'.format(puzzle_name, nkey))
        fn_list = pathlib.Path(fl.screen).glob("edge_batch-*.npz")
        all_edge_from = []
        all_edge_to = []
        for fn in progressbar(fn_list):
            d = matio.load(fn)
            all_edge_from.append(d['EDGE_FROM'])
            all_edge_to.append(d['EDGE_TO'])
        edges_from = util.safe_concatente(all_edge_from, axis=0)
        edges_to = util.safe_concatente(all_edge_to, axis=0)
        assert edges_from.shape[0] == edges_to.shape[0]

        range_dic = {}
        for i, scheme in enumerate(RAW_KEY_PRED_SCHEMES):
            range_dic[scheme] = (bases[i], bases[i+1])
        range_dic['cmb'] = (util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS, nkey)

        for scheme, rtup in range_dic.items():
            util.log(f'[assemble_roots][scheme {scheme}] rtup {rtup}')
            fl.update_scheme(scheme)
            vert_ids = [i for i in range(rtup[0], rtup[1])]
            djs = disjoint_set.DisjointSet(vert_ids)
            for r,c in zip(edges_from, edges_to):
                # util.log(f'[screen_keyconf][{puzzle_name}][scheme {scheme}] union {r} {c}')
                if r >= rtup[1] or r < rtup[0]:
                    continue
                if c >= rtup[1] or c < rtup[0]:
                    continue
                djs.union(r,c)
            cluster = djs.get_cluster()
            util.log("[screen_keyconf][{}][scheme {scheme}] DisjointSet cluster number {nc}".format(puzzle_name, scheme=scheme, nc=len(cluster)))
            screened_index = []

            unit_keys = uw.translate_ompl_to_unit(keys)
            util.log("[screen_keyconf][{}][scheme {scheme}] Screening roots".format(puzzle_name, scheme, scheme=scheme))
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
            util.log("[screen_keyconf][{}][scheme {scheme}] Screened {} roots into {}".format(
                      puzzle_name,
                      rtup[1] - rtup[0] + util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS, screened.shape, scheme=scheme))
            np.savez_compressed(fl.screened_key_fn, KEYQ_OMPL=screened)
            util.ack("[screen_keyconf][scheme {scheme}] Save screened roots {} to {}".format(screened.shape, fl.screened_key_fn, scheme=scheme))

"""
Note: In stage starting from bloomings, we are NOT going to evaluate the schemes that failed to
      predicate any key configuration. Hence valid_puzzle_generator should be used.

      Also the ALGO_VERSION shall be set here
"""
def valid_puzzle_generator(ws, args):
    # ALGO_VERSION = 3
    ALGO_VERSION = 4
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        fl = FileLocations(args, ws, puzzle_name, ALGO_VERSION=ALGO_VERSION)
        key_fn = fl.screened_key_fn
        nkey = matio.load(key_fn)['KEYQ_OMPL'].shape[0]
        if nkey <= util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS:
            util.log(f"[{args.stage}] No {fl.scheme} key configurations predicated for {puzzle_fn}, skipping")
            continue
        yield puzzle_fn, puzzle_name, fl

def blooming(args, ws):
    if args.rerun:
        for puzzle_fn, puzzle_name, fl in valid_puzzle_generator(ws, args):
            _, config = parse_ompl.parse_simple(puzzle_fn)
            key_fn = fl.screened_key_fn
            nkey = matio.load(key_fn)['KEYQ_OMPL'].shape[0]
            bloom_quota = ws.config.getint('Solver', 'PDSBloom')
            if fl.scheme != 'cmb':
                all_keys = matio.load(fl.cmb_screened_key_fn)['KEYQ_OMPL']
                total_quota = bloom_quota * all_keys.shape[0]
                bloom_quota = total_quota // nkey + int(not not (total_quota % nkey))
            for i in progressbar(range(nkey)):
                outfn = join(fl.bloom, f'bloom-from_{i}.npz')
                if os.path.isfile(outfn):
                    continue
                util.log(f'<{args.current_trial}> [blooming][{puzzle_name}] rerun task {i}')
                shell_args = ['python3',
                        'se3solver.py',
                        'solve',
                        '--cdres',
                        config.getfloat('problem', 'collision_resolution', fallback=0.0001),
                        '--replace_istate',
                        f'file={key_fn},key=KEYQ_OMPL,offset={i},size=1,out={fl.bloom}',
                        '--bloom_out',
                        outfn,
                        puzzle_fn,
                        util.RDT_FOREST_ALGORITHM_ID,
                        0.1,
                        '--bloom_limit',
                        bloom_quota]
                util.shell(shell_args)
        return
    if not args.only_wait:
        for puzzle_fn, puzzle_name, fl in valid_puzzle_generator(ws, args):
            _, config = parse_ompl.parse_simple(puzzle_fn)

            util.log('[blooming] scratch {}'.format(fl.rel_bloom))
            key_fn = fl.screened_key_fn
            nkey = matio.load(key_fn)['KEYQ_OMPL'].shape[0]
            bloom_quota = ws.config.getint('Solver', 'PDSBloom')
            if fl.scheme != 'cmb':
                all_keys = matio.load(fl.cmb_screened_key_fn)['KEYQ_OMPL']
                total_quota = bloom_quota * all_keys.shape[0]
                bloom_quota = total_quota // nkey + int(not not (total_quota % nkey))
            condor_job_args = ['se3solver.py',
                    'solve',
                    '--cdres', config.getfloat('problem', 'collision_resolution', fallback=0.0001),
                    '--replace_istate',
                    f'file={key_fn},key=KEYQ_OMPL,offset=$$([$(Process)]),size=1,out={fl.bloom}',
                    '--bloom_out',
                    join(fl.bloom, 'bloom-from_$(Process).npz'),
                    puzzle_fn,
                    util.RDT_FOREST_ALGORITHM_ID,
                    0.1,
                    '--bloom_limit',
                    bloom_quota]
            condor.local_submit(ws,
                                util.PYTHON,
                                iodir_rel=fl.rel_bloom,
                                arguments=condor_job_args,
                                instances=nkey,
                                wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
    if args.no_wait:
        return
    for puzzle_fn, puzzle_name, fl in valid_puzzle_generator(ws, args):
        condor.local_wait(fl.bloom)

def assemble_blooming(args, ws):
    for puzzle_fn, puzzle_name, fl in valid_puzzle_generator(ws, args):
        pds_fn = fl.pds_fn
        Q_list = []
        QE_list = []
        tree_base_list = []
        edge_base_list = []
        INDEX_TO_BLOOM_NO = []
        fn_list = sorted(pathlib.Path(fl.bloom).glob("bloom-from_*.npz"))
        BLOOM_NO_TO_INDEX = np.full((len(fn_list)), -1, dtype=np.int32)
        # fn_list = util.lsv(scratch_dir, prefix="bloom-from_", suffix=".npz")
        tree_base = 0
        edge_base = 0
        for fni, fn in enumerate(progressbar(fn_list)):
            d = matio.load(fn)
            s = d['BLOOM'].shape
            if s[0] == 0:
                continue
            assert s[1] == 7, "{}'s shape is {}".format(fn, s)
            fnstr = str(fn.name)
            assert fnstr.startswith('bloom-from_')
            bloom_idstr = fnstr[len('bloom-from_'):]
            assert bloom_idstr.endswith('.npz')
            bloom_idstr = bloom_idstr[:-len('.npz')]
            assert f"bloom-from_{bloom_idstr}.npz" == fnstr
            bloom_id = int(bloom_idstr)
            BLOOM_NO_TO_INDEX[bloom_id] = len(INDEX_TO_BLOOM_NO)
            INDEX_TO_BLOOM_NO.append(bloom_id)

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
            np.savez_compressed(pds_fn, Q=Q, QF=QF, QB=tree_base_list,
                                QE=QE, QEB=edge_base_list,
                                BLOOM_NO_TO_INDEX=BLOOM_NO_TO_INDEX,
                                INDEX_TO_BLOOM_NO=INDEX_TO_BLOOM_NO)
        else:
            np.savez_compressed(pds_fn, Q=Q, QF=QF, QB=tree_base_list,
                                BLOOM_NO_TO_INDEX=BLOOM_NO_TO_INDEX,
                                INDEX_TO_BLOOM_NO=INDEX_TO_BLOOM_NO)
        util.log('[assemble_blooming] samples stored at {}'.format(pds_fn))

'''
knn_forest:
    Connect forest with prm like algorithm
'''
def pairwise_knn(args, ws):
    if args.rerun:
        for puzzle_fn, puzzle_name, fl in valid_puzzle_generator(ws, args):
            key_fn = fl.screened_key_fn
            keys = matio.load(key_fn)['KEYQ_OMPL']
            nkey = keys.shape[0]
            for i in progressbar(range(nkey)):
                fl.update_task_id(i)
                if os.path.isfile(fl.knn_fn):
                    continue
                util.log(f'\n<{args.current_trial}> [pairwise_knn][{puzzle_name}] rerun task {i}')
                util.shell(['./facade.py',
                    'solve2',
                    '--stage', 'pairwise_knn',
                    '--current_trial', str(args.current_trial),
                    '--puzzle_name', puzzle_name,
                    '--scheme', args.scheme,
                    '--task_id', str(i),
                    ws.local_ws()])
        return
    if args.task_id is None and not args.only_wait:
        for puzzle_fn, puzzle_name, fl in valid_puzzle_generator(ws, args):
            _, config = parse_ompl.parse_simple(puzzle_fn)
            key_fn = fl.screened_key_fn
            keys = matio.load(key_fn)
            condor_job_args = ['./facade.py',
                    'solve2',
                    '--stage', 'pairwise_knn',
                    '--current_trial', str(args.current_trial),
                    '--puzzle_name', puzzle_name,
                    '--scheme', args.scheme,
                    '--task_id', '$(Process)',
                    ws.local_ws()]
            condor.local_submit(ws,
                                util.PYTHON,
                                iodir_rel=fl.rel_knn,
                                arguments=condor_job_args,
                                instances=keys['KEYQ_OMPL'].shape[0],
                                wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
    if args.task_id is not None:
        for puzzle_fn, puzzle_name, fl in valid_puzzle_generator(ws, args):
            solver_args = TmpDriverArgs()
            solver_args.puzzle = puzzle_fn
            rel_bloom = fl.rel_bloom
            solver_args.bloom_dir = ws.local_ws(rel_bloom)
            solver_args.out = fl.knn_fn
            solver_args.knn = 8 # default
            solver_args.algo_version = fl.ALGO_VERSION # algorithm version
            solver_args.subset = np.array([args.task_id], dtype=np.int)
            se3solver.merge_blooming_forest(solver_args)
        return
    if args.no_wait:
        return

    for puzzle_fn, puzzle_name, fl in valid_puzzle_generator(ws, args):
        condor.local_wait(fl.knn)

def assemble_knn(args, ws):
    for puzzle_fn, puzzle_name, fl in valid_puzzle_generator(ws, args):
        ITE_array = [matio.load(fn)['INTER_BLOOMING_TREE_EDGES'] for _,fn in fl.knn_fn_gen]
        ITE = util.safe_concatente(ITE_array, axis=0)
        if ITE.shape[0] != 0:
            dedupITE = np.unique(ITE[:,[0,2]], axis=0)
        else:
            dedupITE = np.array([], dtype=ITE.dtype)
        np.savez_compressed(fl.ibte_fn, INTER_BLOOMING_TREE_EDGES=ITE, DEDUP_INTER_BLOOMING_TREE_EDGES=dedupITE)

VIRTUAL_OPEN_SPACE_NODE = 1j
OPENSPACE_FLAG = 1

def _extract_bound(B, total, i):
    f = B[i]
    t = total if i == B.shape[0] - 1 else B[i+1]
    return f, t

def tree_level_path(Q, QB, QE, QEB, QF, from_fi, from_vi, to_fi, to_vi):
    import networkx as nx
    assert from_fi == to_fi, f'from_fi {from_fi} does not match to_fi {to_fi}'
    q_from, q_to = _extract_bound(QB, Q.shape[0], from_fi)
    from_gvi = q_from + from_vi
    to_gvi = q_from + to_vi
    print(f'from_fi {from_fi} to_fi {to_fi}')
    print(f'from_vi {from_vi}')
    print(f'to_vi {to_vi}')
    print(f'from_gvi {from_gvi}')
    print(f'to_gvi {to_gvi}')
    qe_from, qe_to = _extract_bound(QEB, QE.shape[0], from_fi)
    print(f'NQ {q_to - q_from}')
    print(f'NE {qe_to - qe_from}')
    G = nx.Graph()
    # G.add_nodes_from([i for i in range(q_from, q_to)] + [VIRTUAL_OPEN_SPACE_NODE])
    G.add_nodes_from([i for i in range(q_from, q_to)])
    G.add_edges_from(QE[qe_from:qe_to])
    if to_vi == VIRTUAL_OPEN_SPACE_NODE:
        virtual_edges = []
        to_gvi = None
        for index, flag in enumerate(QF[q_from:q_to]):
            gindex = index + q_from
            '''
            if flag & OPENSPACE_FLAG:
                assert q_from <= gindex
                virtual_edges.append((gindex, VIRTUAL_OPEN_SPACE_NODE))
            '''
            if flag & OPENSPACE_FLAG:
                to_gvi = gindex
                break
        assert to_gvi is not None
        '''
        util.log("virtual_edges {}".format(virtual_edges))
        G.add_edges_from(virtual_edges)
        '''
    '''
    # San check
    for qi in progressbar(range(q_from, q_to)):
        nx.shortest_path(G, from_gvi, qi)
    '''
    ids = np.array(nx.shortest_path(G, from_gvi, to_gvi))
    return ids

def connect_knn(args, ws):
    if args.no_wait:
        return
    algoprefix = f'{args.scheme}-pairwise_knn-'
    for puzzle_fn, puzzle_name, fl in valid_puzzle_generator(ws, args):

        d = matio.load(fl.ibte_fn)
        ITE = d['INTER_BLOOMING_TREE_EDGES']
        dedupITE = d['DEDUP_INTER_BLOOMING_TREE_EDGES']
        util.log(f"[connect_knn] IBTE (shape: {ITE.shape}) loaded from {fl.ibte_fn}")
        if ITE.shape[0] == 0:
            util.warn(f'[connect_knn] Cannot find path for puzzle {puzzle_name} since there is no IBTE (shape {ITE.shape})')
            continue
        pds_fn = fl.pds_fn
        d = matio.load(pds_fn)
        QF = d['QF']
        QB = d['QB']
        """
        We need to be aware of one important differents between bloom-from_*.npz files and the PDS
        file. Some blooming tree may be empty.
        Hence the Bloom No. (denoted by the number in the file name) may not be aligned to
        the index in the PDS file.

        However, the inter_blooming_tree_edges (ITE) uses bloom no. rather than the PDS index.
        Thus the translation is needed.
        """
        BLOOM_NO_TO_INDEX = d['BLOOM_NO_TO_INDEX']
        INDEX_TO_BLOOM_NO = d['INDEX_TO_BLOOM_NO']

        import networkx as nx
        # Forest level path
        G = nx.Graph()
        G.add_nodes_from([i for i in range(len(BLOOM_NO_TO_INDEX))] + [VIRTUAL_OPEN_SPACE_NODE])
        G.add_edges_from(dedupITE)
        """
        # Goal tree is default at open set
        openset = [1]
        virtual_edges = [(1, VIRTUAL_OPEN_SPACE_NODE)]
        """
        openset = []
        virtual_edges = []
        """
        Connect OpenSet trees to VIRTUAL_OPEN_SPACE_NODE
        """
        for tree_index, tree_base in enumerate(QB):
            _, tree_end = _extract_bound(QB, QF.shape[0], tree_index)
            """
            Translate back to bloom no. to be compatitable with ITE
            """
            tree_no = INDEX_TO_BLOOM_NO[tree_index]
            # print(tree_base)
            # print(tree_end)
            for flag in QF[tree_base:tree_end]:
                if flag & OPENSPACE_FLAG:
                    openset.append(tree_no)
                    virtual_edges.append((tree_no, VIRTUAL_OPEN_SPACE_NODE))
                    break
        util.log("OpenSet {}".format(openset))
        G.add_edges_from(virtual_edges)
        try:
            ids = nx.shortest_path(G, 0, VIRTUAL_OPEN_SPACE_NODE)
            util.log('Forest-level shortest path {}'.format(ids))
        except nx.exception.NetworkXNoPath:
            util.warn(f'[connect_knn] Cannot find path for puzzle {puzzle_name}')
            with ws.open_performance_log() as f:
                print(f"<{ws.current_trial}> [solve2][connect_knn][{args.scheme}] FAIL_TO_SOLVE {puzzle_name}", file=f)
            continue
        util.ack('[solve2][connect_knn] forest level path (bloom no.) {}'.format(ids))
        ids = [BLOOM_NO_TO_INDEX[i] for i in ids[:-1]]
        ids.append(VIRTUAL_OPEN_SPACE_NODE)
        util.ack('[solve2][connect_knn] forest level path (PDS index) {}'.format(ids))

        """
        We use bloom no. in forest level path, because ITE uses bloom no.
        In tree level path we uses pds index instead,
        because we tree level data mainly comes from the PDS file.
        """
        ITE_meta = {}
        def insert_ite(from_fi, from_vi, to_fi, to_vi):
            '''
            if from_fi == 0 and to_fi == 78:
                util.log('ite {}'.format(ite))
            '''
            if from_fi not in ITE_meta:
                ITE_meta[from_fi] = {}
            if to_fi not in ITE_meta[from_fi]:
                ITE_meta[from_fi][to_fi] = [(from_vi, to_vi)]
            else:
                ITE_meta[from_fi][to_fi].append((from_vi, to_vi))
        for index, ite in progressbar(enumerate(ITE)):
            from_fi, from_vi, to_fi, to_vi = ite
            from_fi = BLOOM_NO_TO_INDEX[from_fi]
            to_fi = BLOOM_NO_TO_INDEX[to_fi]
            insert_ite(from_fi, from_vi, to_fi, to_vi)
            insert_ite(to_fi, to_vi, from_fi, from_vi)
        # from_fi, to_fi = 0, 82
        # util.log(f'ITE_meta[{from_fi}][{to_fi}] {ITE_meta[from_fi][to_fi]}')
        Q = d['Q']
        QE = d['QE']
        QEB = d['QEB']
        ompl_q = []
        prev_fi = 0
        prev_vi = matio.load(fl.bloom0_fn)['IS_INDICES'][0]
        try:
            for from_fi, to_fi in zip(ids, ids[1:]):
                if to_fi != VIRTUAL_OPEN_SPACE_NODE:
                    from_vi, to_vi = ITE_meta[from_fi][to_fi][0]
                else:
                    from_fi = prev_fi
                    from_vi = VIRTUAL_OPEN_SPACE_NODE
                # if to_vi == VIRTUAL_OPEN_SPACE_NODE:
                #     ids = ids[:-1]
                util.log(f"prev_fi {prev_fi} prev_vi {prev_vi} from_fi {from_fi} from_vi {from_vi}")
                ids = tree_level_path(Q, QB, QE, QEB, QF, prev_fi, prev_vi, from_fi, from_vi)
                q_from = QB[from_fi]
                local_ids = ids - q_from
                util.log(f"Tree {from_fi} IDS {ids}, Local IDS {local_ids}")
                qs = Q[ids, :]
                util.log(f"Shape {qs.shape}")
                ompl_q.append(qs)
                # if to_fi != VIRTUAL_OPEN_SPACE_NODE:
                #   gvi = QB[to_fi] + to_vi
                #   ompl_q.append(Q[gvi:gvi+1,:])
                prev_fi = to_fi
                prev_vi = to_vi
            # ompl_q.append(tree_level_path(Q, QB, QE, QEB, QF, prev_fi, prev_vi, prev_fi, VIRTUAL_OPEN_SPACE_NODE))
        except nx.exception.NetworkXNoPath:
            assert False, "Should not happen. Found forest-level path but no tree-level path."
            continue
        ompl_q = util.safe_concatente(ompl_q, axis=0)
        path_out = fl.path_out_fn
        matio.savetxt(path_out, ompl_q)
        util.ack("Saving OMPL solution of {} to {}".format(puzzle_name, path_out))

        uw = util.create_unit_world(puzzle_fn)
        unit_q = uw.translate_ompl_to_unit(ompl_q)
        sol_out = fl.unit_out_fn
        matio.savetxt(sol_out, unit_q)
        util.ack("Saving UNIT solution of {} to {}".format(puzzle_name, sol_out))

        with ws.open_performance_log() as f:
            print(f"<{ws.current_trial}> [solve2][connect_knn][{args.scheme}] SOLVED {puzzle_name}", file=f)

function_dict = {
        'least_visible_keyconf_fixed': least_visible_keyconf_fixed,
        'assemble_raw_keyconf': assemble_raw_keyconf,
        'screen_keyconf': screen_keyconf,
        'assemble_roots': assemble_roots,
        'blooming' : blooming,
        'assemble_blooming' : assemble_blooming,
        'pairwise_knn': pairwise_knn,
        'assemble_knn': assemble_knn,
        'connect_knn': connect_knn,
}

def setup_parser(subparsers, module_name='solve2'):
    p = subparsers.add_parser(module_name, help='Solve the puzzle with path planner',
                              formatter_class=choice_formatter.Formatter)
    p.add_argument('--stage',
                   choices=list(function_dict.keys()),
                   help='R|Possible stages:\n'+'\n'.join(list(function_dict.keys())),
                   default='',
                   metavar='')
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--no_wait', action='store_true')
    p.add_argument('--task_id', help='Feed $(Process) from HTCondor', type=int, default=None)
    p.add_argument('--puzzle_name', help='Pick a single puzzle to solve (default to all)', default='')
    p.add_argument('--scheme', help='Choose key prediction scheme',
                   choices=KEY_PRED_SCHEMES,
                   required=True)
    p.add_argument('--rerun', action='store_true')
    """
    p.add_argument('--current_trial', help='Trial to solve the puzzle', type=str, default='0')
    p.add_argument('dir', help='Workspace directory')
    """
    util.set_common_arguments(p)
    return p

def run(args):
    if args.stage in function_dict:
        ws = util.create_workspace_from_args(args)
        function_dict[args.stage](args, ws)
        """
        ws = util.Workspace(args.dir)
        for current_trial in util.rangestring_to_list(args.current_trial):
            ws.current_trial = current_trial
            function_dict[args.stage](args, ws)
        """
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
                      'solve2', cmd, auto_retry=auto_retry,
                      with_trial=True,
                      extra_args=extra_args)

def _remote_command_distributed(ws, cmd, extra_args=''):
    for host,_,puzzle_name in ws.condor_host_vs_test_puzzle_generator():
        _remote_command(ws, cmd,
                        alter_host=host,
                        extra_args=extra_args+' --puzzle_name {} --no_wait'.format(puzzle_name))
    for host,_,puzzle_name in ws.condor_host_vs_test_puzzle_generator():
        _remote_command(ws, cmd,
                        alter_host=host,
                        extra_args=extra_args+' --puzzle_name {} --only_wait'.format(puzzle_name))

def _remote_command_auto(ws, cmd, extra_args=''):
    if ws.condor_extra_hosts:
        _remote_command_distributed(ws, cmd, extra_args=extra_args)
    else:
        _remote_command(ws, cmd, extra_args=extra_args)

def remote_least_visible_keyconf_fixed(ws):
    _remote_command_auto(ws, 'least_visible_keyconf_fixed', extra_args='--scheme cmb')

def remote_assemble_raw_keyconf(ws):
    _remote_command_auto(ws, 'assemble_raw_keyconf', extra_args='--scheme cmb')

def remote_screen_keyconf(ws):
    _remote_command_auto(ws, 'screen_keyconf', extra_args='--scheme cmb --no_wait')
    _remote_command_auto(ws, 'screen_keyconf', extra_args='--scheme cmb --only_wait')

def remote_assemble_roots(ws):
    _remote_command(ws, 'assemble_roots', extra_args='--scheme cmb')

class Launcher(object):
    def __init__(self, stage_name, extra_args):
        self._stage_name = str(stage_name)
        self._extra_args = str(extra_args)

    def __call__(self, ws):
        _remote_command(ws, self._stage_name, extra_args=self._extra_args)

def get_schemed_remoter(stage_name, is_async=False):
    ret = []
    if is_async:
        for scheme in KEY_PRED_SCHEMES:
            ret.append((f'{stage_name}_{scheme}_launch', Launcher(stage_name, f'--no_wait --scheme {scheme}')))

        for scheme in KEY_PRED_SCHEMES:
            ret.append((f'{stage_name}_{scheme}_sync', Launcher(stage_name, f'--only_wait --scheme {scheme}')))
    else:
        for scheme in KEY_PRED_SCHEMES:
            ret.append((f'{stage_name}_{scheme}', Launcher(stage_name, f'--scheme {scheme}')))
    return ret

def collect_stages(variant=0):
    if variant in [6]:
        ret = [
                ('least_visible_keyconf_fixed', remote_least_visible_keyconf_fixed),
                ('assemble_raw_keyconf', remote_assemble_raw_keyconf),
                ('screen_keyconf', remote_screen_keyconf),
                ('assemble_roots', remote_assemble_roots)
              ]
        stages = [
                  'blooming',
                  'assemble_blooming',
                  'pairwise_knn',
                  'assemble_knn',
                  'connect_knn',
                ]
        ret += get_schemed_remoter('blooming', is_async=True)
        ret += get_schemed_remoter('assemble_blooming')
        ret += get_schemed_remoter('pairwise_knn', is_async=True)
        ret += get_schemed_remoter('assemble_knn')
        ret += get_schemed_remoter('connect_knn')
    else:
        assert False, f'Solve Pipeline Variant {variant} has not been implemented'
    return ret
