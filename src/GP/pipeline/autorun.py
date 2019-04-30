#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import preprocess_key
from . import preprocess_surface
from . import train
from . import solve
from . import choice_formatter
from . import util

def get_pipeline_stages():
    stages = []
    for m in [preprocess_key, preprocess_surface, train, solve]:
        stages += m.collect_stages()
    stages += [('End', lambda x: util.ack('All pipeline finished'))]
    return stages

def setup_parser(subparsers):
    p = subparsers.add_parser('autorun', help='Run all pipelines automatically',
                              formatter_class=choice_formatter.Formatter)
    p.add_argument('dir', help='Workspace directory')
    pdesc = get_pipeline_stages()
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
    # print('Total Stages: ' + str(len(stage_names)))

def run(args):
    if args.stage is None:
        preprocess_key.autorun(args)
        preprocess_surface.autorun(args)
        train.autorun(args)
        solve.autorun(args)
    else:
        pdesc = get_pipeline_stages()
        cont = None
        till = None
        for index,(k,v) in enumerate(pdesc):
            util.log('[autorun] checking {}'.format(k))
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
        nstage = []
        if args.till:
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
        util.log("[autorun] running the following stages {}".format([k for k,_ in stage_list]))
        for k,v in stage_list:
            util.ack('[{}] starting...'.format(k))
            v(ws)
            util.ack('[{}] finished'.format(k))
        if nstage:
            util.ack('[autorun] Next stage is {}'.format(nstage[0][0]))
