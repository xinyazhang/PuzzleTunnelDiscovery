# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import os
from os.path import join
import sys
import csv
import json
import itertools
import numpy as np
from collections import OrderedDict

import pyse3ompl as plan
from . import util
from . import matio
from . import condor
from .file_locations import FEAT_PRED_SCHEMES, KEY_PRED_SCHEMES, FileLocations

def human_format(num):
    magnitude = 0
    while abs(num) >= 100:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def _dic_add(dic, key, v):
    if v is None:
        return
    if key in dic:
        if isinstance(v, list):
            dic[key] += v
        else:
            dic[key].append(v)
    else:
        if isinstance(v, list):
            dic[key] = v
        else:
            dic[key] = [v]

def _dic_add_path(dic, keys, v):
    if v is None:
        return
    key = keys[0]
    if len(keys) == 1:
        _dic_add(dic, key, v)
        return
    if key not in dic:
        dic[key] = {}
    _dic_add_path(dic[key], keys[1:], v)

def _dic_fetch_path(dic, keys):
    key = keys[0]
    if key not in dic:
        return []
    if len(keys) == 1:
        return dic[key]
    return _dic_fetch_path(dic[key], keys[1:])

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

def knn6_dir(puzzle_name, current_trial):
    return join(util.SOLVER_SCRATCH, puzzle_name, 'pairwise_knn6-{}'.format(current_trial))

_CONDOR_SOLSTAGE_TO_DIR = {
        'estimate_keyconf_clearance' : estimate_keyconf_clearance_dir,
        'screen_keyconf' : screen_keyconf_dir,
        'sample_pds' : sample_pds_dir,
        'forest_rdt' : forest_rdt_dir,
        'forest_rdt_withbt' : forest_rdt_withbt_dir,
        #'knn3' : knn3_dir,
        'knn6' : knn6_dir,
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

class Tabler(object):
    PAPERMAP = OrderedDict({
            'alpha-1.1' : [('alpha-1.1', 'env')],
            'alpha-1.0' : [('alpha-1.0', 'env,rob')],
            'ag'        : [('ag-2', 'env,rob')],
            'aj'        : [('aj,aj-2', 'env,rob')],
            'az'        : [('az', 'env,rob')],
            'double-alpha' : [('doublealpha-1.0', 'env')],
            'claw'      : [('claw-rightbv.dt.tcp', 'env,rob')],
            'duet ring' : [('duet-g1,duet-g2,duet-g4,duet-g9,duet-g9-alternative', 'rob')],
            'duet-g1 grid' : [('duet-g1', 'env')],
            'duet-g2 grid' : [('duet-g2', 'env')],
            'duet-g4 grid' : [('duet-g4', 'env')],
            'duet-g9(a) grid' : [('duet-g9,duet-g9-alternative', 'env')],
            'enigma part 1' : [('enigma', 'env')],
            'enigma part 2' : [('enigma', 'rob')],
            'ABC part AB'   : [('abc_rec2m', 'env')],
            'ABC part C'   : [('abc_rec2m', 'rob')],
            'Key I part 1'   : [('key_1_rec2', 'env')],
            'Key I part 2'   : [('key_1_rec2', 'rob')],
    })
    PAPER_TRANSLATION = {
            'abc_rec2m' : 'ABC',
            'ag-2': 'ag',
            'claw-rightbv.dt.tcp': 'claw',
            'doublealpha-1.0': 'double-alpha',
            'duet-g9-alternative': 'duet-g9a',
            'key_1_rec2': 'Key I',
            'key_2': 'Key II',
            'laby_rec2': 'Laby',
            'mobius': 'Mobius',
    }

class FeatStatTabler(Tabler):
    """
    Variables to list schemes

    Commonly override
    """
    SCHEMES = FEAT_PRED_SCHEMES

    def __init__(self, args):
        self.args = args

    """
    _fl_to_raw_data: generate raw data for each FileLocations object

    Commonly override
    """
    def _fl_to_raw_data(self, fl):
        for geo_type in ['env', 'rob']:
            keyfn = fl.get_feat_pts_fn(geo_type)
            key = fl.feat_npz_key
            yield geo_type, matio.load_safeshape(keyfn, key)[0]

    """
    raw_data_gen: generate raw data from (dir, trial, puzzle_fn) tuples

    Rarely override
    """
    def raw_data_gen(self):
        args = self.args
        trial_list = util.rangestring_to_list(args.trial_range)
        for d in args.dirs:
            ws = util.Workspace(d)
            ws.override_config(args.override_config)
            for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
                for trial in trial_list:
                    ws.current_trial = trial
                    fl = FileLocations(args, ws, puzzle_name, ALGO_VERSION=6)
                    for scheme in self.SCHEMES:
                        fl.update_scheme(scheme)
                        for geo_type, data in self._fl_to_raw_data(fl):
                            yield puzzle_name, scheme, geo_type, data

    """
    Collect (puzzle_name, geo_type, number of feature points/key confs) tuples

    Rarely override
    """
    def collect_raw_data(self):
        dic = {}
        for puzzle_name, scheme, geo_type, nfeat in self.raw_data_gen():
            p = [puzzle_name, scheme, geo_type]
            # print(f'{p} {nfeat}')
            _dic_add_path(dic, p, nfeat)
        return dic

    """
    Helper function used by collect_agg_data

    No need to override since its caller (collect_agg_data) is overrided
    """
    def where_gen(self, where_list):
        for puzzle_name_str, geo_type_str in where_list:
            puzzle_names = puzzle_name_str.split(',')
            geo_types = geo_type_str.split(',')
            for n,t in itertools.product(puzzle_names, geo_types):
                yield n, t

    """
    Aggregate the dict into paper-ready csv

    Commonly override
    """
    def collect_agg_data(self, dic):
        adic = {}
        for name, where_list in self.PAPERMAP.items():
            ad = {}
            for scheme in self.SCHEMES:
                l = []
                for puzzle_name, geo_type in self.where_gen(where_list):
                    p = [puzzle_name, scheme, geo_type]
                    f = _dic_fetch_path(dic, p)
                    # print(f"fetch {p} as {f}")
                    l += f
                ad[f'{scheme}.list'] = l
                ad[f'{scheme}.mean'] = [float(np.mean(l))]
                ad[f'{scheme}.stdev'] = [float(np.std(l))]
            adic[name] = ad
        return adic

    """
    Read header of the table as the first row of matrix

    Commonly override
    """
    def get_matrix_cols(self):
        row = []
        for scheme, stat in itertools.product(self.SCHEMES, ['mean', 'stdev']):
            row += [f'{scheme}.{stat}']
        return row

    """
    Read the matrix item from the aggressive dict

    Commonly override
    """
    def get_matrix_item(self, adic, row_name, col_name):
        p = [row_name, col_name]
        # print(f"fetching {p}")
        f = _dic_fetch_path(adic, p)
        return f

    """
    Translate the aggressive dict into read-to-print matrix

    Rarely override
    """
    def collect_matrix(self, adic):
        matrix = []
        row = ['Puzzle Name'] + self.get_matrix_cols()
        matrix.append(row)
        for name in adic.keys():
            row = [name]
            for col_name in self.get_matrix_cols():
                # print(f"fetch {p} as {f}")
                row += self.get_matrix_item(adic, name, col_name)
            matrix.append(row)
        return matrix

    """
    Run the whole tabler engine, including:
        1. Collect raw data
        2. Aggressive statistical data into dict
        3. Translate dict into ready-for-print matrix.

    Rarely override
    """
    def print(self):
        dic = self.collect_raw_data()
        print(dic)
        adic = self.collect_agg_data(dic)
        print(adic)
        matrix = self.collect_matrix(adic)
        with open(self.args.tex, 'w') as file:
            _print_latex(matrix, float_fmt="{0:.2f}", file=file)

def feat(args):
    tabler = FeatStatTabler(args)
    tabler.print()

class KeyStatTabler(FeatStatTabler):
    SCHEMES = KEY_PRED_SCHEMES

    def __init__(self, args):
        super().__init__(args)

    def _fl_to_raw_data(self, fl):
        v1 = matio.load_safeshape(fl.raw_key_fn, 'KEYQ_OMPL')[0]
        v1 = v1 - util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS if v1 is not None else v1
        v2 = matio.load_safeshape(fl.screened_key_fn, 'KEYQ_OMPL')[0]
        v2 = v2 - util.RDT_FOREST_INIT_AND_GOAL_RESERVATIONS if v2 is not None else v2
        yield 'raw', v1
        yield 'screened', v2

    def collect_agg_data(self, dic):
        adic = {}
        for puzzle_name in dic.keys():
            ad = {}
            for scheme, raw in itertools.product(self.SCHEMES, ['raw', 'screened']):
                p = [puzzle_name, scheme, raw]
                l = _dic_fetch_path(dic, p)
                ad[f'{scheme}.{raw}.list'] = l
                ad[f'{scheme}.{raw}.mean'] = [float(np.mean(l))]
                ad[f'{scheme}.{raw}.stdev'] = [float(np.std(l))]
            if puzzle_name in self.PAPER_TRANSLATION:
                name = self.PAPER_TRANSLATION[puzzle_name]
            else:
                name = puzzle_name
            adic[name] = ad
        return adic

    def get_matrix_cols(self):
        row = []
        for scheme, raw, stat in itertools.product(self.SCHEMES, ['screened'], ['mean', 'stdev']):
            row += [f'{scheme}.{raw}.{stat}']
        return row

def keyq(args):
    tabler = KeyStatTabler(args)
    tabler.print()

class SolveStatTabler(FeatStatTabler):
    BASELINES =  ['rdt', 'rdtc', 'prm']
    BASELINE_IDS = {
            'rdt' : plan.PLANNER_RDT,
            'rdtc' : plan.PLANNER_RDT_CONNECT,
            'prm' : plan.PLANNER_PRM,
            }
    SCHEMES = KEY_PRED_SCHEMES + ['mcheck'] + BASELINES

    def __init__(self, args):
        super().__init__(args)
        self._collected_baseline = []

    def _fl_to_raw_data(self, fl):
        if fl.scheme in KEY_PRED_SCHEMES:
            if os.path.isfile(fl.unit_out_fn):
                yield 'solve', 1
            elif os.path.isfile(fl.performance_log):
                yield 'solve', 0
            else:
                yield 'solve', None
        elif fl.scheme == 'mcheck':
            fl.update_scheme('cmb') # Choose read mcheck data from 'cmb' scheme
            try:
                cur_ec = 0
                cur_ec_time = 0
                for i, bloom_fn in fl.bloom_fn_gen:
                    # util.warn(f'loading {bloom_fn}')
                    d = matio.load(bloom_fn)
                    cur_ec += int(d['PF_LOG_MCHECK_N'])
                    cur_ec_time += float(d['PF_LOG_MCHECK_T'])
                for i, knn_fn in fl.knn_fn_gen:
                    # util.warn(f'loading {knn_fn}')
                    d = matio.load(knn_fn)
                    cur_ec += int(d['PF_LOG_MCHECK_N'])
                    cur_ec_time += float(d['PF_LOG_MCHECK_T'])
            except Exception as e:
                util.warn(f'[{fl.puzzle_name}][trial {fl.trial}] is incomplete, some file is missing')
                yield 'solve', None
                return
            yield 'solve', cur_ec
        else:
            baseline_trial = self.args.baseline_trial
            if baseline_trial is None:
                yield 'solve', None
            baseline_dir = fl.get_baseline_dir(planner_id=self.BASELINE_IDS[fl.scheme],
                                               trial_id=baseline_trial)
            baseline_dir = str(baseline_dir)
            if baseline_dir in self._collected_baseline:
                yield 'solve', None # Already collected
                return
            print(f'collecting {baseline_dir}')
            # print(f'{self._collected_baseline}')
            # for c in self._collected_baseline:
            #     assert c != baseline_dir
            self._collected_baseline.append(baseline_dir)
            solution_list = []
            for fn in fl.get_baseline_files(baseline_dir):
                try:
                    if matio.load(fn, key='FLAG_IS_COMPLETE') != 0:
                        solution_list.append(1)
                    else:
                        solution_list.append(0)
                except:
                    util.log(f'{fn} cannot be read')
                    solution_list.append(None)
            yield 'solve', solution_list # TODO: add baseline support

    # TODO: merge this with KeyStatTabler's collect_agg_data
    def collect_agg_data(self, dic):
        adic = {}
        for puzzle_name in dic.keys():
            ad = {}
            for scheme in self.SCHEMES:
                raw = 'solve'
                p = [puzzle_name, scheme, raw]
                l = _dic_fetch_path(dic, p)
                ad[f'{scheme}.solve.list'] = l
                ad[f'{scheme}.solve.solved'] = [np.sum([1 if e == 1 else 0 for e in l], dtype=int)]
                ad[f'{scheme}.solve.total'] = [np.sum([1 if e is not None else 0 for e in l], dtype=int)]
            if puzzle_name in self.PAPER_TRANSLATION:
                name = self.PAPER_TRANSLATION[puzzle_name]
            else:
                name = puzzle_name
            adic[name] = ad
        return adic

    def get_matrix_cols(self):
        return self.SCHEMES

    def get_matrix_item(self, adic, row_name, col_name):
        if col_name == 'mcheck':
            p = [row_name, f'{col_name}.solve.list']
            list_with_none = _dic_fetch_path(adic, p)
            list_wo_none = []
            for e in list_with_none:
                if e is not None:
                    list_wo_none.append(e)
            if len(list_wo_none) == 0:
                return [f'TBD']
            return [f'{human_format(np.mean(list_wo_none))}']
        p = [row_name, f'{col_name}.solve.solved']
        # print(f"fetching {p}")
        solved = _dic_fetch_path(adic, p)
        solved_int = solved[0] if solved else 0
        p = [row_name, f'{col_name}.solve.total']
        total = _dic_fetch_path(adic, p)
        total_int = total[0] if total else 0
        return [f'{100*solved_int/total_int:.1f}'] if total_int > 0 else ['TBD']

class WiderSolveStatTabler(SolveStatTabler):
    BASELINES = [
            'RRT-Connect',
            'RRT',
            'RDT',
            'RDT-Connect',
            'PRM',
            'BITstar',
            'EST',
            'LBKPIECE1',
            'BKPIECE1',
            'KPIECE1',
            'PDST',
            'SBL',
            'STRIDE',
            'FMT',
    ]

    BASELINE_IDS = {
            'RRT-Connect': plan.PLANNER_RRT_CONNECT,
            'RRT':         plan.PLANNER_RRT,
            'RDT':         plan.PLANNER_RDT,
            'RDT-Connect': plan.PLANNER_RDT_CONNECT,
            'PRM':         plan.PLANNER_PRM,
            'BITstar':     plan.PLANNER_BITstar,
            'EST':         plan.PLANNER_EST,
            'LBKPIECE1':   plan.PLANNER_LBKPIECE1,
            'BKPIECE1':    plan.PLANNER_BKPIECE1,
            'KPIECE1':     plan.PLANNER_KPIECE1,
            'PDST':        plan.PLANNER_PDST,
            'SBL':         plan.PLANNER_SBL,
            'STRIDE':      plan.PLANNER_STRIDE,
            'FMT':         plan.PLANNER_FMT,
            }
    SCHEMES = BASELINES

    def __init__(self, args):
        super().__init__(args)
        self._collected_baseline = []

    def get_matrix_cols(self):
        return self.BASELINES

    '''
    Override this to transpose this matrix
    '''
    def collect_matrix(self, adic):
        matrix = []
        testing_puzzles = ['alpha-1.0', 'duet-g2']
        row = ['Planner Name'] + testing_puzzles
        matrix.append(row)
        for planner_name in self.BASELINES:
            row = [planner_name]
            for puzzle_name in testing_puzzles:
                # print(f"fetch {p} as {f}")
                row += self.get_matrix_item(adic, puzzle_name, planner_name)
            matrix.append(row)
        return matrix

def solve(args):
    if args.all_planners:
        tabler = WiderSolveStatTabler(args)
    else:
        tabler = SolveStatTabler(args)
    tabler.print()

class PlannerTimingTabler(FeatStatTabler):
    # SCHEMES = KEY_PRED_SCHEMES
    SCHEMES = ['cmb']
    """
    PF_KEYS is the 'geo_type' of FeatStatTabler
    """
    PF_KEYS = ['PF_LOG_PLAN_T',
            # 'PF_LOG_MCHECK_N',
            'PF_LOG_MCHECK_T',
            # 'PF_LOG_DCHECK_N',
            'PF_LOG_KNN_QUERY_T',
            'PF_LOG_KNN_DELETE_T',
            ]

    def __init__(self, args):
        super().__init__(args)

    def _fl_to_raw_data(self, fl):
        """
        Data from blooming tree
        """
        for i, fn in fl.bloom_fn_gen:
            d = matio.safeopen(fn)
            for k in self.PF_KEYS:
                if k in d:
                    yield k, d[k]
        for i, fn in fl.knn_fn_gen:
            d = matio.safeopen(fn)
            for k in self.PF_KEYS:
                if k in d:
                    yield k, d[k]

    # TODO: merge this with KeyStatTabler's collect_agg_data
    def collect_agg_data(self, dic):
        adic = {}
        for puzzle_name in dic.keys():
            ad = {}
            for scheme, pf_key in itertools.product(self.SCHEMES, self.PF_KEYS):
                p = [puzzle_name, scheme, pf_key]
                l = _dic_fetch_path(dic, p)
                ad[f'{scheme}.{pf_key}.list'] = l
                ad[f'{scheme}.{pf_key}.mean'] = [float(np.mean(l))]
                ad[f'{scheme}.{pf_key}.stdev'] = [float(np.std(l))]
                ad[f'{scheme}.{pf_key}.sum'] = [float(np.sum(l))]
            if puzzle_name in self.PAPER_TRANSLATION:
                name = self.PAPER_TRANSLATION[puzzle_name]
            else:
                name = puzzle_name
            adic[name] = ad
        return adic

    def get_matrix_cols(self):
        return self.PF_KEYS

    def get_matrix_item(self, adic, row_name, col_name):
        p = [row_name, f'cmb.{col_name}.sum']
        # print(f"fetching {p}")
        f = _dic_fetch_path(adic, p)
        return [f'{f[0] / 1e3 / 3600:.2f} hrs']

def timing_the_planner(args):
    tabler = PlannerTimingTabler(args)
    tabler.print()

class CondorHours(FeatStatTabler):
    # SCHEMES = KEY_PRED_SCHEMES
    SCHEMES = ['cmb']
    """
    PF_KEYS is the 'geo_type' of FeatStatTabler
    """
    PF_KEYS = ['Clearance Estimation', 'Screening', 'Blooming', 'KNN' ]
    # PF_KEYS = ['Blooming', 'KNN' ]

    def __init__(self, args):
        super().__init__(args)

    def _fl_to_raw_data(self, fl):
        from . import condor
        yield 'Clearance Estimation', condor.query_last_cputime_from_log(join(fl.clearance, 'log'), translate_to_msecs=True)
        """
        Data from blooming tree
        """
        total = 0
        for i, fn in fl.bloom_fn_gen:
            try:
                d = matio.load(fn)
                total += d['PF_LOG_PLAN_T']
            except:
                pass
        yield 'Blooming', total
        if fl.has_screening:
            yield 'Screening', condor.query_last_cputime_from_log(join(fl.screen, 'log'), translate_to_msecs=True)
        else:
            yield 'Screening', None
        total = 0
        for i, fn in fl.knn_fn_gen:
            try:
                d = matio.load(fn)
                total += d['PF_LOG_PLAN_T']
            except:
                pass
        yield 'KNN', total

    # TODO: merge this with KeyStatTabler's collect_agg_data
    def collect_agg_data(self, dic):
        adic = {}
        for puzzle_name in dic.keys():
            ad = {}
            for scheme, pf_key in itertools.product(self.SCHEMES, self.PF_KEYS):
                p = [puzzle_name, scheme, pf_key]
                l = _dic_fetch_path(dic, p)
                ad[f'{scheme}.{pf_key}.list'] = l
                ad[f'{scheme}.{pf_key}.mean'] = [float(np.mean(l))]
                ad[f'{scheme}.{pf_key}.stdev'] = [float(np.std(l))]
                ad[f'{scheme}.{pf_key}.sum'] = [float(np.sum(l))]
            if puzzle_name in self.PAPER_TRANSLATION:
                name = self.PAPER_TRANSLATION[puzzle_name]
            else:
                name = puzzle_name
            adic[name] = ad
        return adic

    def get_matrix_cols(self):
        return self.PF_KEYS

    def get_matrix_item(self, adic, row_name, col_name):
        p = [row_name, f'cmb.{col_name}.mean']
        # print(f"fetching {p}")
        f = _dic_fetch_path(adic, p)
        return [f'{f[0] / 1e3 / 3600:.2f} hrs']

def condor_hours(args):
    tabler = CondorHours(args)
    tabler.print()

class WallclockBreakdownTimer(FeatStatTabler):
    # SCHEMES = KEY_PRED_SCHEMES
    SCHEMES = ['cmb']

    def __init__(self, args):
        super().__init__(args)
        self.stage_set = set()
        self.stage_list = []

    def _fl_to_raw_data(self, fl):
        logfn = fl.performance_log
        if not os.path.isfile(logfn):
            yield None, None
            return
        util.log(f'reading {logfn}')
        BREAKER = ' cost '
        cost_dic = {}
        with open(logfn, 'r') as f:
            for line in f:
                loc = line.find(BREAKER)
                if loc < 0:
                    continue
                cost_str = line[loc+len(BREAKER):].strip()
                line = line.replace('[', ' ')
                line = line.replace(']', ' ')
                split = line.split()
                stage_name = split[0]
                day_break = cost_str.find('+')
                hr_break = cost_str.find(':')
                mi_break = cost_str.find(':', hr_break+1)
                sec_break = cost_str.find(':', mi_break+1)
                days = int(cost_str[:day_break])
                hrs = int(cost_str[day_break+1:hr_break])
                mins = int(cost_str[hr_break+1:mi_break])
                sec = float(cost_str[mi_break+1:])
                cost_in_msec = 1e3 * (3600*24*days + 3600 * hrs+ 60 * mins + sec)
                if stage_name not in self.stage_set:
                    self.stage_set.add(stage_name)
                    self.stage_list.append(stage_name)
                # util.log(f'stage {stage_name} cost {cost_str} ({cost_in_msec} in msec)')
                cost_dic[stage_name] = cost_in_msec
        for stage_name in self.stage_list:
            # util.log(f'yielding {stage_name} cost_dic[stage_name]')
            yield stage_name, cost_dic[stage_name] if stage_name in cost_dic else None

    def collect_agg_data(self, dic):
        adic = {}
        for puzzle_name in dic.keys():
            ad = {}
            for scheme, pf_key in itertools.product(self.SCHEMES, self.stage_list):
                p = [puzzle_name, scheme, pf_key]
                l = _dic_fetch_path(dic, p)
                ad[f'{scheme}.{pf_key}.list'] = l
                ad[f'{scheme}.{pf_key}.mean'] = [float(np.mean(l))]
                ad[f'{scheme}.{pf_key}.stdev'] = [float(np.std(l))]
                ad[f'{scheme}.{pf_key}.sum'] = [float(np.sum(l))]
            if puzzle_name in self.PAPER_TRANSLATION:
                name = self.PAPER_TRANSLATION[puzzle_name]
            else:
                name = puzzle_name
            adic[name] = ad
        return adic

    def get_matrix_cols(self):
        return self.stage_list

    def get_matrix_item(self, adic, row_name, col_name):
        p = [row_name, f'cmb.{col_name}.mean']
        # print(f"fetching {p}")
        f = _dic_fetch_path(adic, p)
        return [f'{f[0] / 1e3 / 3600:.2f} hrs']

def breakdown(args):
    breaker = WallclockBreakdownTimer(args)
    breaker.print()

    """
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
    """

def _print_latex(matrix, float_fmt="{0:.2f}", file=None, align=4):
    print(matrix)
    f = sys.stdout if file is None else file
    print(f)
    ncol = len(matrix[0])
    for i in range(len(matrix)):
        assert len(matrix[i]) == ncol, f'len(matrix[{i}]) = {len(matrix[i])} != {ncol}, details: {matrix[i]}'
    print(f"ncol {ncol}")
    def fmt(item):
        if isinstance(item, float):
            return float_fmt.format(item)
        return item
    widths = [1 + max([len(fmt(matrix[i][j])) for i in range(len(matrix))]) for j in range(ncol)]
    print(widths)
    awidths = [ w + (4 - w % align) for w in widths ]
    print(awidths)
    for i in range(len(matrix)):
        end = ""
        for j in range(ncol):
            print(end, end='', file=f)
            out = fmt(matrix[i][j])
            out = out.ljust(widths[j])
            print(out , end='', file=f)
            end = "&"
        print("\\\\", file=f)

function_dict = {
        'conclude' : conclude,
        'breakdown' : breakdown,
        'condor_breakdown' : condor_breakdown,
        'condor_ppbreakdown' : condor_ppbreakdown,
        'feat' : feat,
        'keyq' : keyq,
        'solve' : solve,
        'timing_the_planner' : timing_the_planner,
        'condor_hours' : condor_hours,
}

def setup_parser(subparsers):
    sp = subparsers.add_parser('stats', help='Various Statistic Tools.')
    toolp = sp.add_subparsers(dest='tool_name', help='Name of Tool')

    p = toolp.add_parser('conclude', help='Show the execution statistics')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--puzzle_name', help='Only show one specific testing puzzle', default='')
    p.add_argument('--type', help='Choose what kind of info to output', default='detail', choices=['detail', 'stat'])
    p.add_argument('--out', help='Output CSV file', default='1.csv')
    p.add_argument('dirs', help='Workspace directory', nargs='+')

    p = toolp.add_parser('breakdown', help='Show the per-stage runtime')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--puzzle_name', help='Only show one specific testing puzzle', default='')
    p.add_argument('--override_config', help='Override workspace config', default='')
    p.add_argument('--type', help='Choose what kind of info to output', default='detail', choices=['detail', 'stat'])
    p.add_argument('--tex', help='Output paper-ready table body TeX', default='b.tex')
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
    p.add_argument('dirs', help='Archived workspace directory, must include condor log files', nargs='+')

    p = toolp.add_parser('feat', help='Collect the feature points into a table')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--tex', help='Output paper-ready table body TeX', default='feat.tex')
    p.add_argument('--override_config', help='Override workspace config', default='')
    p.add_argument('dirs', help='Archived workspace directory.', nargs='+')

    p = toolp.add_parser('keyq', help='Collect the key configurations into a table')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--tex', help='Output paper-ready table body TeX', default='keyq.tex')
    p.add_argument('--override_config', help='Override workspace config', default='')
    p.add_argument('dirs', help='Archived workspace directory.', nargs='+')

    p = toolp.add_parser('solve', help='Collect chance to solve into a table')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--baseline_trial', help='Trial of running baseline', type=int, default=None)
    p.add_argument('--tex', help='Output paper-ready table body TeX', default='sol.tex')
    p.add_argument('--override_config', help='Override workspace config', default='')
    p.add_argument('--all_planners', help='Use wider range of planners', action='store_true')
    p.add_argument('dirs', help='Archived workspace directory.', nargs='+')

    p = toolp.add_parser('timing_the_planner', help='Collect timing of the blooming and forest connection into a table')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--tex', help='Output paper-ready table body TeX', default='planner_pf.tex')
    p.add_argument('dirs', help='Archived workspace directory.', nargs='+')

    p = toolp.add_parser('condor_hours', help='Show CPU hours spent by HTCondor')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--puzzle_name', help='Only show one specific testing puzzle', default='')
    p.add_argument('--tex', help='Output tex file', default='c.tex')
    p.add_argument('--override_config', help='Override workspace config', default='')
    p.add_argument('dirs', help='Archived workspace directory, must include condor log files',
                   nargs='+')

def run(args):
    function_dict[args.tool_name](args)
