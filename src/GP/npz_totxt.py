#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

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
