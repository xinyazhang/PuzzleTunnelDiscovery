#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
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
    for m in [train, keyconf, solve2]:
        stages += m.collect_stages(variant=10)
    stages += [('End', lambda x: util.ack('All pipeline finished'))]
    return stages

def setup_parser(subparsers):
    autorun.setup_autorun_parser(subparsers, 'autorun10', _get_pipeline_stages(),
                                 helptext='NN only pipeline')

def run(args):
    if args.stage is None:
        args.stage = 'Begin'
        args.till = 'End'
    if args.nn_profile != "":
        util.fatal("do NOT pass --nn_profile to autorun7. This auto pipeline will set nn_profile automatically")
        exit()
    autorun.run_pipeline(_get_pipeline_stages(), args)
