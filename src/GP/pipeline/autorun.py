#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import preprocess_key
import preprocess_surface
import train
import solve

def setup_parser(subparsers):
    p = subparsers.add('runall', help='Run all pipelines automatically')
    p.add_argument('dir', help='Workspace directory')

def run(args):
    preprocess_key.autorun(args)
    preprocess_surface.autorun(args)
    train.autorun(args)
    solve.autorun(args)
