#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np
import os
import sys
import argparse

def head(fns):
    for fn in fns:
        d = np.load(fn)
        print("==> {} <==".format(fn))
        for k,v in d.items(): # Python 3 syntax
            print("{}: type {} shape {}".format(k, v.dtype, v.shape))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('files', help='NPZ File', nargs='+')
    args = parser.parse_args()
    head(args.files)

if __name__ == '__main__':
    main()
