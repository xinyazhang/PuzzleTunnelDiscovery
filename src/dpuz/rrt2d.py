#!/usr/bin/env python3
# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import argparse
from collections import namedtuple
import pycutec2

NAN = float('nan')
INVALID_POINT = np.array([NAN, NAN])
l2norm = np.linalg.norm

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('geo', help='Geometry NPZ file')
    p.add_argument('--I', help='Initial state', nargs=2, type=float, default=[0,0])
    p.add_argument('--G', help='Goal state', nargs=2, type=float, default=[NAN, NAN])
    p.add_argument('--max', help='Max iterations', type=int, default=5)
    p.add_argument('--num', help='Total number of trees', type=int, default=1)
    p.add_argument('--bb_min', help='Bounding Box (min)', nargs=2, type=float, default=[-1,-1])
    p.add_argument('--bb_max', help='Bounding Box (max)', nargs=2, type=float, default=[ 1, 1])
    p.add_argument('--out', help='Output NPZ file for the RRT', default=None)

    args = p.parse_args()
    '''
    Python List -> numpy array
    '''
    for attr in ['I', 'G', 'bb_min', 'bb_max']:
        setattr(args, attr, np.array(getattr(args, attr)))

    return args

Tree = namedtuple('Tree', ['V', 'E', 'SOL'])
Tree.__new__.__defaults__ = (None,) * len(Tree._fields)

class RRT2D(object):
    def __init__(self, args):
        self._args = args
        self._trees = []
        self._union_tree = Tree()
        self.load(args.geo)
        self._no_goal = np.isnan(args.G).any()

    def load(self, geo_fn):
        d = np.load(geo_fn)
        self._mesh = Tree(V=d['V'], E=d['E'])

    def collide(self, q, p):
        return pycutec2.line_segment_intersect_with_mesh(p, q, self._mesh.V, self._mesh.E)

    def sampler(self, args):
        return np.random.uniform(low=args.bb_min, high=args.bb_max)

    def run_once(self, args):
        V = np.full(shape=[args.max + 1, 2], fill_value=np.nan, dtype=np.float64)
        E = np.full(shape=[args.max + 1], fill_value=-1, dtype=np.int)
        no_goal = np.isnan(args.G).any()
        # print(V.shape)
        V[0,:] = args.I
        E[0] = -1
        solved = False
        for i in range(1, args.max+1):
            q = self.sampler(args)
            distances = l2norm(V[:i] - q, axis=1)
            close_i = np.nanargmin(distances)
            hit,_,_,_ = self.collide(q, V[close_i])
            if hit:
                E[i] = -1
                V[i,:] = INVALID_POINT
            else:
                E[i] = close_i
                V[i,:] = q
                if not no_goal:
                    hit,_,_,_ = self.collide(q, args.G)
                    if not hit:
                        solved = True
                        V = np.concatenate((V[:i+1], [args.G]), axis=0)
                        E = np.concatenate((E[:i+1], [i]), axis=0)
                        break

        return Tree(V=V, E=E, SOL=solved)

    def run(self):
        args = self._args
        if self._no_goal:
            V = np.full(shape=[args.num, args.max + 1, 2], fill_value=np.nan,
                        dtype=np.float64)
            E = np.full(shape=[args.num, args.max + 1], fill_value=-1,
                        dtype=np.int)
            SOL = np.full(shape=[args.num], fill_value=0,
                          dtype=np.int)
            for i in range(args.num):
                t = self.run_once(args)
                V[i] = t.V
                E[i] = t.E
                SOL[i] = t.SOL
                # self._trees.append(t)
            self._union_tree = Tree(V=V, E=E, SOL=SOL)
        else:
            for i in range(args.num):
                self._trees.append(self.run_once(args))

    def save(self):
        args = self._args
        dic = { 'I' : args.I,
                'G' : args.G,
                'N' : args.max,
                'NT': args.num,
                'BB_MAX': args.bb_max,
                'BB_MIN': args.bb_min
               }
        if self._no_goal:
            dic['V'] = self._union_tree.V
            dic['E'] = self._union_tree.E
            dic['SOL'] = self._union_tree.SOL
        else:
            for i in range(args.num):
                dic[f'{i}.V'] = self._trees[i].V
                dic[f'{i}.E'] = self._trees[i].E
                dic[f'{i}.SOL'] = self._trees[i].SOL
        if args.out is None:
            print(dic)
            return
        np.savez_compressed(args.out, **dic)

def main():
    args = parse_args()
    print(args)
    rrt = RRT2D(args)
    rrt.run()
    rrt.save()

if __name__ == '__main__':
    main()
