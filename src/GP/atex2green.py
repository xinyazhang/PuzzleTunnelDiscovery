#!/usr/bin/env python3
# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later

import numpy as np
import argparse
import pathlib
from imageio import imwrite

def sigmoid(x):
    from scipy.special import expit
    return expit(x)

def show_stat(x):
    print("min {} max {} avg {} median {} stddev {}".format(np.min(x), np.max(x), np.mean(x), np.median(x), np.std(x)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", help=".npz atex files", nargs='+')
    p.add_argument("--clipmin", help="clip the value before saving", type=float, default=None)
    p.add_argument("--clipmax", help="clip the value before saving", type=float, default=None)
    p.add_argument("--binary_thresh", help="Turn the texture into binary", type=float, default=None)
    p.add_argument("--sigmoid", help="Apply sigmoid()", action='store_true')
    args = p.parse_args()
    for fn in args.files:
        d = np.load(fn)
        if 'ATEX' not in d:
            continue
        atex = d['ATEX'].astype(np.float32)
        if args.sigmoid:
            nz = np.nonzero(atex)
            show_stat(atex[nz])
            atex[nz] = sigmoid(atex[nz])
            show_stat(atex)
        if args.clipmin is not None or args.clipmax is not None:
            np.clip(atex, a_min=args.clipmin, a_max=args.clipmax, out=atex)
        if args.binary_thresh is not None:
            old_atex = np.copy(atex)
            atex[old_atex >= args.binary_thresh] = 1.0
            atex[old_atex < args.binary_thresh] = 0.0
        gatex = np.zeros(shape=(atex.shape[0], atex.shape[1], 3))
        ma = np.max(atex)
        mi = np.min(atex)
        natex = atex - mi
        if ma - mi > 0:
            natex /= (ma - mi)
        gatex[:,:,1] = natex
        pn = pathlib.Path(fn)
        ofn = pn.with_suffix('.png')
        imwrite(ofn, gatex)
        print("{} => {}".format(fn, ofn))


if __name__ == '__main__':
    main()
