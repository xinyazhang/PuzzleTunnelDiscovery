#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import matio
from . import util
import os
from os.path import join, isdir, isfile
import pathlib

FEAT_PRED_SCHEMES = ['ge', 'nt' ]
RAW_KEY_PRED_SCHEMES = ['ge', 'nt', 'nn']
KEY_PRED_SCHEMES = RAW_KEY_PRED_SCHEMES + ['cmb']
SCHEME_TO_FMT = {
        'ge' : util.GERATIO_KEY_FMT,
        'nt' : util.NOTCH_KEY_FMT,
        'nn' : util.NEURAL_KEY_FMT,
        'cmb' : util.COMBINED_KEY_FMT
        }

SCHEME_FEAT_FMT = {
        'ge' : util.GERATIO_POINT_FMT,
        'nt' : util.NOTCH_POINT_FMT,
        }

SCHEME_FEAT_NPZ_KEY = {
        'ge' : 'KEY_POINT_AMBIENT',
        'nt' : 'NOTCH_POINT_AMBIENT',
}

class FileLocations(object):
    def __init__(self, args, ws, puzzle_name, scheme=None, ALGO_VERSION=6):
        self._args = args
        self._ws = ws
        self._puzzle_name = puzzle_name
        self._scheme = args.scheme if hasattr(args, 'scheme') else ''
        self._task_id = None
        self.ALGO_VERSION = ALGO_VERSION

    def update_scheme(self, scheme):
        self._scheme = scheme

    def update_task_id(self, task_id):
        self._task_id = task_id

    @property
    def scheme(self):
        return self._scheme

    @property
    def scheme_prefix(self):
        if self.scheme:
            return f'{self.scheme}-'
        else:
            return ''

    @property
    def trial(self):
        return self._ws.current_trial

    @property
    def task_id(self):
        if self._task_id is not None:
            return self._task_id
        assert self._args.task_id is not None
        return self._args.task_id

    '''
    Convention:
        rel_ prefix: relative path from ws.dir
        _fn suffix: file name
        no prefix/suffix: absolute directory
    '''

    @property
    def rel_clearance(self):
        return join(util.SOLVER_SCRATCH, self._puzzle_name,
                    util.KEYCONF_CLEARANCE_DIR, str(self.trial))

    @property
    def clearance(self):
        return self._ws.local_ws(self.rel_clearance)

    @property
    def downsampled_key_fn(self):
        return self._ws.keyconf_file_from_fmt(self._puzzle_name,
                                              util.NEURAL_KEY_FMT)
    def get_feat_pts_fn(self, geo_type):
        fmt = SCHEME_FEAT_FMT[self.scheme]
        return self._ws.local_ws(util.TESTING_DIR,
                                 self._puzzle_name,
                                 fmt.format(geo_type=geo_type, trial=self.trial))
    @property
    def feat_npz_key(self):
        return SCHEME_FEAT_NPZ_KEY[self.scheme]

    @property
    def raw_key_fn_gen(self):
        def gen():
            for scheme in RAW_KEY_PRED_SCHEMES:
                yield scheme, self._ws.keyconf_file_from_fmt(self._puzzle_name,
                                                             SCHEME_TO_FMT[scheme])
        return gen()

    @property
    def assembled_raw_key_fn(self):
        return self._ws.keyconf_file_from_fmt(self._puzzle_name, 'all_raw_keyconf-{trial}.npz')

    @property
    def raw_key_fn(self):
        if self.scheme == 'cmb':
            return self.assembled_raw_key_fn
        return self._ws.keyconf_file_from_fmt(self._puzzle_name,
                                              SCHEME_TO_FMT[self.scheme])
    @property
    def cmb_raw_key_fn(self):
        return self.assembled_raw_key_fn

    @property
    def rel_screen(self):
        return join(util.SOLVER_SCRATCH, self._puzzle_name,
                    f'screen-{self.trial}')

    @property
    def screen(self):
        return self._ws.local_ws(self.rel_screen)

    @property
    def screened_key_fn(self):
        conf = self._ws.config.getboolean('Solver', 'EnableKeyConfScreening', fallback=True)
        print(f'Solver.EnableKeyConfScreening == {conf}')
        if conf == False:
            return self.raw_key_fn
        return self._ws.keyconf_file_from_fmt(self._puzzle_name,
                                              'screened_'+SCHEME_TO_FMT[self.scheme])

    @property
    def cmb_screened_key_fn(self):
        if self._ws.config.getboolean('Solver', 'EnableKeyConfScreening', fallback=True) == False:
            return self.cmb_raw_key_fn

        return self._ws.keyconf_file_from_fmt(self._puzzle_name,
                                              'screened_'+SCHEME_TO_FMT['cmb'])

    @property
    def rel_pds(self):
        return join(util.SOLVER_SCRATCH,
                    self._puzzle_name,
                    f'{self.scheme_prefix}{util.PDS_SUBDIR}')

    @property
    def pds(self):
        ret = self._ws.local_ws(self.rel_pds)
        os.makedirs(ret, exist_ok=True)
        return ret

    @property
    def rel_bloom(self):
        return join(self.rel_pds, f'bloom-{self.trial}')

    @property
    def bloom(self):
        return self._ws.local_ws(self.rel_bloom)

    @property
    def bloom_fn(self):
        return join(self.bloom, f'bloom-from_{self.task_id}.npz')

    @property
    def bloom0_fn(self):
        return join(self.bloom, 'bloom-from_0.npz')

    @property
    def bloom_fn_gen(self):
        keys = matio.safeload(self.screened_key_fn, key='KEYQ_OMPL')
        NTree = keys.shape[0]
        def gen():
            for i in range(NTree):
                yield i, join(self.bloom, f'bloom-from_{i}.npz')
        return gen()

    @property
    def pds_fn(self):
        fn ='{}.npz'.format(self.trial)
        return join(self.pds, fn)

    @property
    def rel_knn(self):
        return join(util.SOLVER_SCRATCH, self._puzzle_name, f'{self.scheme}-knn{self.ALGO_VERSION}-{self.trial}')

    @property
    def knn(self):
        return self._ws.local_ws(self.rel_knn)

    @property
    def knn_fn(self):
        return join(self.knn, f'pairwise_knn_edges-{self.task_id}.npz')

    @property
    def knn_fn_gen(self):
        keys = matio.safeload(self.screened_key_fn, key='KEYQ_OMPL')
        NTree = keys.shape[0]
        def gen():
            for i in range(NTree):
                yield i, join(self.knn, f'pairwise_knn_edges-{i}.npz')
        return gen()

    @property
    def ibte_fn(self):
        return join(self.knn, 'inter_blooming_tree_edges.npz')

    @property
    def path_out_fn(self):
        return join(self.knn, self.scheme_prefix + 'path.txt')

    @property
    def unit_out_fn(self):
        return self._ws.solution_file(self._puzzle_name, type_name=self.scheme_prefix+'unit')

    @property
    def vanilla_out_fn(self):
        return self._ws.solution_file(self._puzzle_name, type_name=self.scheme_prefix+'vanilla')

    @property
    def performance_log(self):
        return self._ws.local_ws(util.PERFORMANCE_LOG_DIR, 'log.{}'.format(self.trial))
