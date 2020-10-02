#!/usr/bin/env python3
# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
# -*- coding: utf-8 -*-

from . import preprocess_key
from . import preprocess_surface
from . import train
from . import keyconf
from . import solve
from . import choice_formatter
from . import util

########################################
# Functions specific for autorun2 (or more)
########################################

def run_pipeline(ppl_stages, args):
    pdesc = ppl_stages
    cont = None
    till = None
    pdesc_dic = {}
    for index,(k,v) in enumerate(pdesc):
        # util.log(f'[{args.command}] checking {k}')
        pdesc_dic[k] = v
        if k == args.stage:
            if cont is not None:
                raise RuntimeError("Duplicated key name {} in the pipeline".format(k))
            cont = index
        if args.till is not None and k == args.till:
            if cont is None:
                raise RuntimeError("--till specified a stage BEFORE --stage")
            till = index + 1
    assert cont is not None
    ws = util.Workspace(args.dir)
    ws.nn_profile = args.nn_profile
    if args.condor_host:
        ws.override_condor_host(args.condor_host)
    ws.override_config(args.override_config)
    nstage = []
    if args.stage_list:
        stage_list = []
        for sname in args.stage_list:
            stage_list.append((sname, pdesc_dic[sname]))
    elif args.till:
        stage_list = pdesc[cont:till]
        if till is not None:
            nstage = pdesc[till:][0:1] # still works if till is out of range/None, or pdesc[till:] is empty
    else:
        stage_list = pdesc[cont:cont+1]
        nstage = pdesc[cont+1:cont+2]
    keys = [k for k,_ in stage_list]
    if None in keys:
        util.warn("[NOTE] Pipeline is broken")
        raise RuntimeError("Pipeline is broken, cannot autorun with --till")
    util.log("[{}] running the following stages {}".format(args.command, [k for k,_ in stage_list]))
    if args.current_trial is not None:
        trials = util.rangestring_to_list(args.current_trial)
    else:
        trials = [None]
    for trial in trials:
        ws.current_trial = trial
        ws.timekeeper_start(args.command)
        with ws.open_performance_log() as f:
            print(f"[{args.command}] TRIAL {trial}", file=f)
            print(f"[{args.command}][arguments] {args}", file=f)
            print(f"[{args.command}][options] {ws.config_as_dict}", file=f)
        with open(ws.local_ws(util.PERFORMANCE_LOG_DIR, 'active_config.{}'.format(ws.current_trial)), 'w') as f:
            ws.config.write(f)
        for k,v in stage_list:
            ws.timekeeper_start(k)
            util.ack('<{}> [{}] starting...'.format(ws.current_trial, k))
            v(ws)
            util.ack('<{}> [{}] finished'.format(ws.current_trial, k))
            ws.timekeeper_finish(k)
        if nstage:
            util.ack('[{}] Next stage is {}'.format(args.command, nstage[0][0]))
        ws.timekeeper_finish(args.command)
        with ws.open_performance_log() as f:
            print(f"[{args.command}] END_OF_TRIAL {trial}", file=f)

def setup_autorun_parser(subparsers, name, pdesc, helptext='Run all pipelines automatically'):
    p = subparsers.add_parser(name, help=helptext,
                              formatter_class=choice_formatter.Formatter)
    p.add_argument('dir', help='Workspace directory')
    stage_names = [ str(k) for k,v in pdesc ]
    # print(stage_names)
    stage_name_str = "\n".join(stage_names)
    # print(stage_name_str)
    p.add_argument('--stage',
                   help='R|Start from one of the ' + str(len(stage_names)) + ' stages in the pipeline. List of pipeline stages\n'+stage_name_str,
                   choices=stage_names,
                   default=None,
                   metavar='')
    p.add_argument('--till', help='Continue to run until the given stage',
                   choices=stage_names,
                   default=None,
                   metavar='')
    p.add_argument('--stage_list', help='Only run these stages in the given order. Overrides --stage and --till',
                   choices=stage_names,
                   default=None,
                   nargs='*',
                   metavar='')
    p.add_argument('--current_trial', help='Trial to solve the puzzle', type=str, default=None)
    p.add_argument('--nn_profile', help='NN profile', default='')
    p.add_argument('--condor_host', help='Override the CondorHost option provided by config File in workspace', type=str, default=None)
    p.add_argument('--override_config', help='Override configurations by config file in workspace. Syntax: SECTION.OPTION=VALUE. Separated by semicolon (;)',
                   type=str, default=None)
    # print('Total Stages: ' + str(len(stage_names)))


########################################
# Functions specific for autorun
########################################

def _get_pipeline_stages():
    stages = []
    stages += [('Begin', lambda x: util.ack('Starting the pipeline'))]
    for m in [preprocess_key, preprocess_surface, train]:
        stages += m.collect_stages()
    stages += [('End', lambda x: util.ack('All pipeline finished'))]
    return stages

def setup_parser(subparsers):
    setup_autorun_parser(subparsers, 'autorun', _get_pipeline_stages(),
            helptext='NN training pipeline')

def run(args):
    if args.stage is None:
        args.stage = 'Begin'
        args.till = 'End'
    run_pipeline(_get_pipeline_stages(), args)
