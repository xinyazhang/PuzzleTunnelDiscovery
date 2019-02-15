#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse

def totxt(fn):
    d = np.load(fn)
    for k in d.keys():
        return d[k]

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out', help='Output txt File', required=True)
    parser.add_argument('file', help='NPZ File')
    args = parser.parse_args()
    a = totxt(args.file)
    np.savetxt(args.out, a, fmt='%.17g')

if __name__ == '__main__':
    main()
