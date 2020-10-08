#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np
import os
import sys

def mean(fns):
    dsum = {}
    for fn in fns:
        d = np.load(fn)
        for k,v in d.items(): # Python 3 syntax
            if k not in dsum:
                dsum[k] = v
            else:
                dsum[k] += v
    denom = float(len(fns))
    for k,v in dsum.items():
        dsum[k] /= denom
    return dsum

def main(files):
    ifns = files[:-1]
    ofn = files[-1]
    dic = mean(ifns)
    np.savez(ofn, **dic)

if __name__ == '__main__':
    main(sys.argv[1:])
