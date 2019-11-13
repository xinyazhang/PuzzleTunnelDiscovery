import os
import sys
import csv
import json
import itertools
import numpy as np
from collections import OrderedDict

from . import util
from . import matio
from . import condor
from .file_locations import FEAT_PRED_SCHEMES, KEY_PRED_SCHEMES, FileLocations

def _dic_add(dic, key, v):
    if v is None:
        return
    if key in dic:
        dic[key].append(v)
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

class Tabler(object):
    PAPERMAP = OrderedDict({
            'alpha-1.1' : [('alpha-1.1', 'env')],
            'alpha-1.0' : [('alpha-1.0', 'env,rob')],
            'ag'        : [('ag-2', 'env,rob')],
            'aj'        : [('aj,aj-2', 'env,rob')],
            'az'        : [('az', 'env,rob')],
            'duet ring' : [('duet-g1,duet-g2,duet-g4,duet-g9,duet-g9-alternative', 'rob')],
            'duet-g1 grid' : [('duet-g1', 'env')],
            'duet-g2 grid' : [('duet-g2', 'env')],
            'duet-g4 grid' : [('duet-g4', 'env')],
            'duet-g9(a) grid' : [('duet-g9,duet-g9-alternative', 'env')],
            'claw'      : [('claw-rightbv.dt.tcp', 'env,rob')],
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
    SCHEMES = FEAT_PRED_SCHEMES

    def __init__(self, args):
        self.args = args

    def _fl_to_raw_data(self, fl):
        for geo_type in ['env', 'rob']:
            keyfn = fl.get_feat_pts_fn(geo_type)
            key = fl.feat_npz_key
            yield geo_type, matio.load_safeshape(keyfn, key)[0]

    def raw_data_gen(self):
        args = self.args
        trial_list = util.rangestring_to_list(args.trial_range)
        for d in args.dirs:
            ws = util.Workspace(d)
            for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
                for trial in trial_list:
                    ws.current_trial = trial
                    fl = FileLocations(args, ws, puzzle_name)
                    for scheme in self.SCHEMES:
                        fl.update_scheme(scheme)
                        for geo_type, data in self._fl_to_raw_data(fl):
                            yield puzzle_name, scheme, geo_type, data

    """
    Collect (puzzle_name, geo_type, number of feature points/key confs) tuples
    """
    def collect_raw_data(self):
        dic = {}
        for puzzle_name, scheme, geo_type, nfeat in self.raw_data_gen():
            p = [puzzle_name, scheme, geo_type]
            # print(f'{p} {nfeat}')
            _dic_add_path(dic, p, nfeat)
        return dic

    def where_gen(self, where_list):
        for puzzle_name_str, geo_type_str in where_list:
            puzzle_names = puzzle_name_str.split(',')
            geo_types = geo_type_str.split(',')
            for n,t in itertools.product(puzzle_names, geo_types):
                yield n, t

    """
    Aggregrate the dict into paper-ready csv
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

    def get_matrix_cols(self):
        row = []
        for scheme, stat in itertools.product(self.SCHEMES, ['mean', 'stdev']):
            row += [f'{scheme}.{stat}']
        return row

    def get_matrix_item(self, adic, row_name, col_name):
        p = [row_name, col_name]
        # print(f"fetching {p}")
        f = _dic_fetch_path(adic, p)
        return f

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
    SCHEMES = KEY_PRED_SCHEMES + BASELINES

    def __init__(self, args):
        super().__init__(args)

    def _fl_to_raw_data(self, fl):
        if fl.scheme in KEY_PRED_SCHEMES:
            if os.path.isfile(fl.unit_out_fn):
                yield 'solve', 1
            elif os.path.isfile(fl.performance_log):
                yield 'solve', 0
            else:
                yield 'solve', None
        else:
            yield 'solve', None # TODO: add baseline support

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
        row = []
        for scheme in self.SCHEMES:
            row += [f'{scheme}']
        return row

    def get_matrix_item(self, adic, row_name, col_name):
        p = [row_name, f'{col_name}.solve.solved']
        # print(f"fetching {p}")
        solved = _dic_fetch_path(adic, p)
        solved_int = solved[0] if solved else 0
        p = [row_name, f'{col_name}.solve.total']
        total = _dic_fetch_path(adic, p)
        total_int = total[0] if total else 0
        return [f'{solved_int}/{total_int}'] if total_int > 0 else ['TBD']

def solve(args):
    tabler = SolveStatTabler(args)
    tabler.print()

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
    p.add_argument('dirs', help='Archived workspace directory, must include condor log files', nargs='+')

    p = toolp.add_parser('feat', help='Collect the feature points into a table')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--tex', help='Output paper-ready table body TeX', default='feat.tex')
    p.add_argument('dirs', help='Archived workspace directory.', nargs='+')

    p = toolp.add_parser('keyq', help='Collect the key configurations into a table')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--tex', help='Output paper-ready table body TeX', default='keyq.tex')
    p.add_argument('dirs', help='Archived workspace directory.', nargs='+')

    p = toolp.add_parser('solve', help='Collect chance to solve into a table')
    p.add_argument('--trial_range', help='range of trials', type=str, required=True)
    p.add_argument('--tex', help='Output paper-ready table body TeX', default='sol.tex')
    p.add_argument('dirs', help='Archived workspace directory.', nargs='+')

def run(args):
    function_dict[args.tool_name](args)
