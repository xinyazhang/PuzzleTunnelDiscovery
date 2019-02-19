#!/usr/bin/env python

class DisjointSet:

    def __init__(self, init_arr):
        self._parents = {}
        if init_arr:
            for item in init_arr:
                self._parents[item] = item

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

    def union(self, elem1, elem2):
        if elem1 == elem2:
            return
        # TODO: more advanced algorithm ?
        elem1 = self.find(elem1)
        elem2 = self.find(elem2)
        assert elem1 is not None and elem2 is not None
        self._parents[elem2] = elem1

    def get_roots(self):
        roots = []
        for key in self._parents:
            value = self._parents[key]
            if value == key:
                roots.append(key)
        return roots

if __name__ == '__main__':
    import argparse
    import numpy as np
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
    else:
        print("Unknown format for file {}".format(args.efile))
        exit()
    vset = [i for i in range(args.n)]
    djs = DisjointSet(vset)
    print(pairs.shape)
    # pairs = np.unique(pairs, axis=0)
    for e in progressbar(pairs):
        [r,c] = e
        djs.union(r,c)
    print(djs.get_roots())
    cluster = dict()
    for v in vset:
        r = djs.find(v)
        if r not in cluster:
            cluster[r] = [v]
        else:
            cluster[r].append(v)
        if djs.find(0) == djs.find(1):
            print("Early terminate: 0 and 1 connected")
            break
    print(cluster)
    print("Root of 0 {}".format(djs.find(0)))
    print("Root of 1 {}".format(djs.find(1)))
