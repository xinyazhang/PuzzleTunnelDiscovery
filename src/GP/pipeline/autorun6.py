#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import train
from . import keyconf
from . import geometrik2
from . import solve2
from . import choice_formatter
from . import util
from . import autorun

def _get_pipeline_stages():
    stages = []
    stages += [('Begin', lambda x: util.ack('Starting the pipeline'))]
    for m in [train, keyconf, geometrik2, solve2]:
        stages += m.collect_stages(variant=6)
    stages += [('End', lambda x: util.ack('All pipeline finished'))]
    return stages

def setup_parser(subparsers):
    autorun.setup_autorun_parser(subparsers, 'autorun6', _get_pipeline_stages(),
                                 helptext='GK+NN combined pipeline')

def run(args):
    if args.stage is None:
        args.stage = 'Begin'
        args.till = 'End'
    autorun.run_pipeline(_get_pipeline_stages(), args)
