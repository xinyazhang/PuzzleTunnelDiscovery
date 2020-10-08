#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np
import os
import sys
import argparse

def cat(fns):
    dlist = {}
    for fn in fns:
        d = np.load(fn)
        for k,v in d.items(): # Python 3 syntax
            if k not in dlist:
                dlist[k] = [v]
            else:
                dlist[k].append(v)
    dcat = {}
    for k,v in dlist.items():
        dcat[k] = np.concatenate(dlist[k])
    return dcat

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out', help='Output NPZ File', required=True)
    parser.add_argument('files', help='NPZ File', nargs='+')
    args = parser.parse_args()
    dic = cat(args.files)
    np.savez(args.out, **dic)

if __name__ == '__main__':
    main()
