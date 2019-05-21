#!/usr/bin/env python3

import numpy as np
import argparse
import pathlib
from imageio import imwrite

def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", help=".npz atex files", nargs='+')
    p.add_argument("--clipmin", help="clip the value before saving", type=float, default=None)
    p.add_argument("--clipmax", help="clip the value before saving", type=float, default=None)
    args = p.parse_args()
    for fn in args.files:
        d = np.load(fn)
        if 'ATEX' not in d:
            continue
        atex = d['ATEX'].astype(np.float32)
        np.clip(atex, a_min=args.clipmin, a_max=args.clipmax, out=atex)
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
