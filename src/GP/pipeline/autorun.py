#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import preprocess_key
from . import preprocess_surface
from . import train
from . import solve
from . import choice_formatter

def get_pipeline_stages():
    stages = []
    for m in [preprocess_key, preprocess_surface, train, solve]:
        stages += m.collect_stages()
    return stages

def setup_parser(subparsers):
    p = subparsers.add_parser('runall', help='Run all pipelines automatically',
                              formatter_class=choice_formatter.Formatter)
    p.add_argument('dir', help='Workspace directory')
    pdesc = get_pipeline_stages()
    stage_names = [ str(k) for k,v in pdesc ]
    # print(stage_names)
    stage_name_str = "\n".join(stage_names)
    # print(stage_name_str)
    p.add_argument('--cont',
                   help='R|Continue from one of the ' + str(len(stage_names)) + ' stages in the pipeline. List of pipeline stages\n'+stage_name_str,
                   choices=stage_names,
                   default=None,
                   metavar='')
    # print('Total Stages: ' + str(len(stage_names)))

def run(args):
    if args.cont is None:
        preprocess_key.autorun(args)
        preprocess_surface.autorun(args)
        train.autorun(args)
        solve.autorun(args)
    else:
        pdesc = get_pipeline_stages()
        cont = None
        for index,(k,v) in enumerate(pdesc):
            if k == args.cont:
                if cont is not None:
                    raise RuntimeError("Duplicated key name {} in the pipeline".format(k))
                cont = index
            if k is None and v is None and cont is not None:
                raise RuntimeError("Pipeline is broken, cannot continue from stage {}".format(args.cnt))
        assert cont is not None
        ws = util.Workspace(args.dir)
        for k,v in pdesc[cont:]:
            v(ws)
