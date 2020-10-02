#!/usr/bin/env python3
# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later

from base import *

def setup_parser(subparsers):
    show_parser = subparsers.add_parser("show", help='Show the number of tunnel vertices')

def run(args):
    ti = TaskInterface(args)
    print("# of tunnel vertices is {}".format(len(ti._get_tunnel_v())))
