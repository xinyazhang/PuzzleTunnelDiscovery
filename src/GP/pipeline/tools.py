#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isdir, isfile
import copy

from . import util
from . import matio
from . import condor

def read_roots(args):
    uw = util.create_unit_world(args.puzzle_fn)
    ompl_q = matio.load(args.roots)[args.roots_key]
    unit_q = uw.translate_ompl_to_unit(ompl_q)
    matio.savetxt(args.out, unit_q)

function_dict = {
        'read_roots' : read_roots
}

def setup_parser(subparsers):
    sp = subparsers.add_parser('tools', help='Various Tools.')
    toolp = sp.add_subparsers(dest='tool_name', help='Name of Tool')
    p = toolp.add_parser('read_roots', help='Dump roots of the forest to text file')
    p.add_argument('--roots_key', help='NPZ file of roots', default='KEYQ_OMPL')
    p.add_argument('puzzle_fn', help='OMPL config')
    p.add_argument('roots', help='NPZ file of roots')
    p.add_argument('out', help='output txt file')

def run(args):
    function_dict[args.tool_name](args)
