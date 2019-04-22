#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import preprocess_key
import preprocess_surface
import train
import solve

def get_pipeline_stages():
    stages = []
    for m in [preprocess_key, preprocess_surface, train, solve]:
        stages += m.collect_stages()
    return stages

def setup_parser(subparsers, pdesc):
    p = subparsers.add('runall', help='Run all pipelines automatically')
    p.add_argument('dir', help='Workspace directory')
    pdesc = get_pipeline_stages()
    stage_names = [ k for k,v in pdesc ]
    p.add_argument('--cont', help='Continue from the given stage', choices=stage_names, default=None)

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
                cont = index
                break
        assert cont is not None
        ws = util.Workspace(args.dir)
        for k,v in pdesc[cont:]:
            v(ws)
