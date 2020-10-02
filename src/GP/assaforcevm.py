#!/usr/bin/env python2
# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later

'''
Assemble Visibility Matrix from condor_saforce.py -> condor_visibility_mc.py
'''

import os
import sys
sys.path.append(os.getcwd())
import json

import numpy as np

def usage():
    print('''
Usage: assaforcevm.py <Task file> <Visibility Matrix Dir> <Output file>
''')

'''
    Task file lists the saforce.py outputs,
    vmdir stores the visibility matrix files from saforce.py outputs
'''
def fn_gen(taskfile, vmdir):
    with open(taskfile) as f:
        tasks = json.load(f)
    for task in tasks:
        infn = task[0]
        safn = task[3]
        fn_base = os.path.basename(safn)
        vmfn = '{}/{}'.format(vmdir, fn_base)
        yield infn, safn, vmfn

def main():
    if sys.argv[1] == '-h':
        usage()
        return
    vmdir = sys.argv[2]
    asfn = sys.argv[3]
    assert os.path.isdir(vmdir), "Argument 2 {} must be a directory".format(asfn)
    assert not os.path.exists(asfn), "Cannot overwrite existing file {}".format(asfn)
    visstat = []
    tocrqs = []
    crqs = []
    q0end = 0
    q1end = 0
    for infn,safn,vmfn in fn_gen(sys.argv[1], vmdir):
        if not os.path.exists(vmfn):
            print("Skipping unfinished VM {}".format(vmfn))
            continue
        d = np.load(vmfn)
        visstat += d['VMFrag'].sum(axis=-1).tolist()
        d = np.load(safn)
        crqs += d['ALL_RQS'].tolist()
        d = np.load(infn)
        tocrqs += d['TOCRQS'].tolist()
    np.savez(asfn, VISSTAT=visstat, CRQS=crqs, TOCRQS=tocrqs)

if __name__ == '__main__':
    main()
