#!/usr/bin/env python2

import numpy as np
from scipy.misc import imread
import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('img', help='Input image file in PNG format', nargs=None, type=str)
    parser.add_argument('npz', help='Output Atex file in NPZ format', nargs=None, type=str)
    args = parser.parse_args()
    img = imread(args.img)
    atex = img[...,1].astype(np.float32)
    np.savez(args.npz, ATEX=atex)

if __name__ == '__main__':
    main()
