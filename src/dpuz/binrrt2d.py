#!/usr/bin/env python3

# import os
# import sys
import numpy as np
import argparse

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('npz_files', help='RRT2D NPZ file', nargs='+')
    p.add_argument('--res', help='Resolution', nargs=2, default=[1024,1024], type=int)
    p.add_argument('--out', help='Binning output', default=None)
    args = p.parse_args()
    return args

class Bin2d(object):

    def __init__(self, args):
        self._args = args
        self._tree_is_unioned = None
        self.load(args)

    def load(self, args):
        samples = []
        self.bb_min = None
        self.bb_max = None
        for fn in args.npz_files:
            d = np.load(fn)
            if self.bb_min is None:
                self.bb_min = d['BB_MIN']
                self.bb_max = d['BB_MAX']
            unioned_tree = True if 'SOL' in d else (False if '0.V' in d else None)
            if unioned_tree is None:
                print(f'skipping unknonw file {fn}')
                continue
            if self._tree_is_unioned is None:
                self._tree_is_unioned = unioned_tree
            else:
                assert self._tree_is_unioned == unioned_tree, 'Cannot combined unioned tree with non unioned tree'
            if unioned_tree:
                V = d['V']
                samples.append(V)
                # print(f'{fn} {V.shape}')
            else:
                for ent in d:
                    if ent.endswith('.V'):
                        a = d[ent][1:-1, :]
                        samples.append(a[~np.isnan(a).any(axis=1)])
        self._V = np.concatenate(samples, axis=0)

    def binning(self):
        print(f'bb_min {self.bb_min}')
        print(f'bb_max {self.bb_max}')
        args = self._args
        if self._tree_is_unioned:
            Hs = []
            N = int(self._V.shape[1])
            accurH = None
            for i in range(1, N):
                X = self._V[:,i,0]
                Y = self._V[:,i,1]
                X = X[~np.isnan(X)]
                Y = Y[~np.isnan(Y)]
                H,self._EX, self._EY = np.histogram2d(X,
                                                      Y,
                                                      bins=args.res,
                                                      range=np.transpose([self.bb_min, self.bb_max]))
                accurH = H if accurH is None else accurH + H
                Hs.append(accurH)
            self._H = np.array(Hs)
        else:
            self._H, self._EX, self._EY = np.histogram2d(self._V[:,0],
                                                         self._V[:,1],
                                                         bins=args.res,
                                                         range=np.transpose([self.bb_min, self.bb_max])
                                                         )

    def save(self):
        args = self._args
        if args.out is None:
            print(self._H)
            print(np.nonzero(self._H))
            print(self._EX)
            print(self._EY)
            return
        np.savez_compressed(args.out, H=self._H, EX=self._EX, EY=self._EY)

def main():
    args = parse_args()
    b = Bin2d(args)
    b.binning()
    b.save()

if __name__ == '__main__':
    main()
