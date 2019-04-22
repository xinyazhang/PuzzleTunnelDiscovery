#!/usr/bin/env python3

from base import *

def setup_parser(subparsers):
    show_parser = subparsers.add_parser("show", help='Show the number of tunnel vertices')

def run(args):
    ti = TaskInterface(args)
    print("# of tunnel vertices is {}".format(len(ti._get_tunnel_v())))
