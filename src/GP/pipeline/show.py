#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

from base import *

def setup_parser(subparsers):
    show_parser = subparsers.add_parser("show", help='Show the number of tunnel vertices')

def run(args):
    ti = TaskInterface(args)
    print("# of tunnel vertices is {}".format(len(ti._get_tunnel_v())))
