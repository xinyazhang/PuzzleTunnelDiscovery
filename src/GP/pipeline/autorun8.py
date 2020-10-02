#!/usr/bin/env python3
# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
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
    for m in [geometrik2, solve2]:
        stages += m.collect_stages(variant=8)
    stages += [('End', lambda x: util.ack('All pipeline finished'))]
    return stages

def setup_parser(subparsers):
    autorun.setup_autorun_parser(subparsers, 'autorun8', _get_pipeline_stages(),
                                 helptext='GER only pipeline')

def run(args):
    if args.stage is None:
        args.stage = 'Begin'
        args.till = 'End'
    if args.nn_profile != "":
        util.fatal("do NOT pass --nn_profile to autorun8. This auto pipeline will set nn_profile automatically")
        exit()
    autorun.run_pipeline(_get_pipeline_stages(), args)
