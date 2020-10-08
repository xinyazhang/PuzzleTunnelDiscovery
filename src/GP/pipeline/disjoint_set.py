#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
# -*- coding: utf-8 -*-

class DisjointSet:

    def __init__(self, init_arr):
        self._parents = {}
        self._parents_uncompressed = {}
        if init_arr:
            for item in init_arr:
                self._parents[item] = item
                self._parents_uncompressed[item] = item

    def find(self, elem):
        if elem not in self._parents:
            return None
        path = []
        while self._parents[elem] != elem:
            path.append(elem)
            elem = self._parents[elem]
        # path compression
        for e in path[1:]:
            self._parents[e] = elem
        return elem

    def find_path(self, elem):
        if elem not in self._parents_uncompressed:
            return []
        path = []
        while self._parents_uncompressed[elem] != elem:
            path.append(elem)
            elem = self._parents_uncompressed[elem]
        path.append(elem)
        return path

    def union(self, elem1, elem2):
        if elem1 == elem2:
            return
        # TODO: more advanced algorithm ?
        elem1 = self.find(elem1)
        elem2 = self.find(elem2)
        assert elem1 is not None and elem2 is not None
        self._parents[elem2] = elem1
        self._parents_uncompressed[elem2] = elem1

    def get_roots(self):
        roots = []
        for key in self._parents:
            value = self._parents[key]
            if value == key:
                roots.append(key)
        return roots

    def get_cluster(self):
        cluster = dict()
        for v in self._parents:
            r = self.find(v)
            if r not in cluster:
                cluster[r] = [v]
            else:
                cluster[r].append(v)
        return cluster

if __name__ == '__main__':
    import argparse
    import numpy as np
    import h5py
    from scipy.io import loadmat,savemat
    from progressbar import progressbar

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('n', help='number of vertices', type=int)
    parser.add_argument('efile', help='edge file')
    args = parser.parse_args()
    if args.efile.endswith('.txt'):
        pairs = np.loadtxt(args.efile, dtype=np.int32, delimiter=",")
    elif args.efile.endswith('.mat'):
        pairs = loadmat(args.efile)['E']
    elif args.efile.endswith('.hdf5'):
        pairs = h5py.File(args.efile, 'r')['E'][:]
    else:
        print("Unknown format for file {}".format(args.efile))
        exit()
    OPENSPACE_NODE = -3
    vset = [i for i in range(args.n)] + [OPENSPACE_NODE]
    djs = DisjointSet(vset)
    print(pairs.shape)
    # pairs = np.unique(pairs, axis=0)
    for e in progressbar(pairs):
        [r,c] = e[0], e[1]
        djs.union(r, c)
        if djs.find(0) == djs.find(1):
            print("Early terminate: 0 and 1 connected")
            break
    print(djs.get_roots())
    cluster = djs.get_cluster()
    print(cluster)
    print("Root of 0 {} Size {}".format(djs.find(0), len(cluster[djs.find(0)])))
    print("Path from 0 {}".format(djs.find_path(0)))
    print("Root of 1 {} Size {}".format(djs.find(1), len(cluster[djs.find(1)])))
    print("Path from 1 {}".format(djs.find_path(1)))
