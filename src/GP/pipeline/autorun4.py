#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import train
from . import keyconf
from . import geometrik
from . import solve
from . import choice_formatter
from . import util
from . import autorun

def _get_pipeline_stages():
    stages = []
    stages += [('Begin', lambda x: util.ack('Starting the pipeline'))]
    for m in [train, keyconf, geometrik, solve]:
        stages += m.collect_stages(variant=4)
    stages += [('End', lambda x: util.ack('All pipeline finished'))]
    return stages

def setup_parser(subparsers):
    autorun.setup_autorun_parser(subparsers, 'autorun4', _get_pipeline_stages())

def run(args):
    if args.stage is None:
        args.stage = 'Begin'
        args.till = 'End'
    autorun.run_pipeline(_get_pipeline_stages(), args)
