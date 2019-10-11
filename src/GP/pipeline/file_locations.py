#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import matio
from . import util
import os
from os.path import join, isdir, isfile

RAW_KEY_PRED_SCHEMES = ['ge', 'nt', 'nn']
KEY_PRED_SCHEMES = RAW_KEY_PRED_SCHEMES + ['cmb']
SCHEME_TO_FMT = {
        'ge' : util.GERATIO_KEY_FMT,
        'nt' : util.NOTCH_KEY_FMT,
        'nn' : util.NEURAL_KEY_FMT,
        'cmb' : util.COMBINED_KEY_FMT
        }

class FileLocations(object):
    def __init__(self, args, ws, puzzle_name, scheme=None):
        self._args = args
        self._ws = ws
        self._puzzle_name = puzzle_name
        self._scheme = args.scheme if hasattr(args, 'scheme') else ''

    def update_scheme(self, scheme):
        self._scheme = scheme

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
        return self._ws.keyconf_file_from_fmt(self._puzzle_name,
                                              SCHEME_TO_FMT[self.scheme])
    @property
    def rel_screen(self):
        return join(util.SOLVER_SCRATCH, self._puzzle_name,
                    f'screen-{self.trial}')

    @property
    def screen(self):
        return self._ws.local_ws(self.rel_screen)

    @property
    def screened_key_fn(self):
        return self._ws.keyconf_file_from_fmt(self._puzzle_name,
                                              'screened_'+SCHEME_TO_FMT[self.scheme])

    @property
    def cmb_screened_key_fn(self):
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
    def pds_fn(self):
        fn ='{}.npz'.format(self.trial)
        return join(self.pds, fn)

    @property
    def rel_knn(self):
        return join(util.SOLVER_SCRATCH, self._puzzle_name, f'{self.scheme}-knn3-{self.trial}')

    @property
    def knn(self):
        return self._ws.local_ws(self.rel_knn)

    @property
    def knn_fn(self):
        return join(self.knn, f'pairwise_knn_edges-{self.task_id}.npz')

    @property
    def knn_fn_gen(self):
        keys = matio.load(self.screened_key_fn)
        NTree = keys['KEYQ_OMPL'].shape[0]
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
