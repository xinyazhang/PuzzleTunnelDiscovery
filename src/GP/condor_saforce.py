#!/usr/bin/env python2
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

import os
import sys
import json
import numpy as np
import saforce

def usage():
    print('''Collision resolution through force defined by surface area, distributed version for HTCondor.
Usage: condor_saforce.py <command> [<args>]

Commands:
1. define
    condor_saforce.py define <[npz files]>
    Setup the taskset definition from input files. The definition is printed to stdout.
    The output should be redirected to a file for `part` command to use
2. part
    condor_saforce.py part <taskset definition file> <tasks per process> <task partition file>
    Partition the tasks in the taskset definition, and print the results to task partition file.
    This also prints the number of tasks to stdout
3. run
    condor_saforce.py run <task partition file> <task index>

# Input:
    A set of .npz files given to `define` command.

# Output:
    For each INPUT.npz, a set of corresponding INPUT-task-#.npz will be defineerated by `run` command.
    This naming scheme is implemented by `part` command.

# Post-processing

1. Use condor_visibility_mc.py to calculate visibility of defineerated samples
    - condor_visibility_mc.py requires the task partition file outputed by `part` command
2. After 1, use assaforcevm.py to assembly the visibility matrix and correponding samples
''')

def define():
    cfg = []
    for fn in sys.argv[2:]:
        d = np.load(fn)
        cfg.append((fn, len(d['TOCRQS'])))
    print(json.dumps(cfg))

def part():
    cfgfile = sys.argv[2]
    gran = int(sys.argv[3])
    taskfile = sys.argv[4]
    with open(cfgfile) as f:
        cfg = json.load(f)
    tasks = []
    total = 0
    for (fn,n) in cfg:
        fn_dir = os.path.dirname(fn)
        fn_base = os.path.basename(fn)
        fn_bare = os.path.splitext(fn_base)[0]
        file_index = 0
        for i in range(0, n, gran):
            ofn = '{}/{}-task-{}.npz'.format(fn_dir, fn_bare, file_index) if fn_dir else '{}-task-{}.npz'.format(fn_bare, file_index)
            task = (fn, i, i+gran, ofn)
            tasks.append(task)
            file_index += 1
            total += 1
    with open(taskfile, 'w') as f:
        json.dump(tasks, fp=f)
    print("Total tasks {}".format(total))

def run():
    taskfile = sys.argv[2]

    with open(taskfile) as f:
        tasks = json.load(f)
    task = tasks[int(sys.argv[3])]
    saforce.solve(task[0], task[1], task[2], task[3])

def main():
    cmd = sys.argv[1]
    if cmd in ['-h', 'help']:
        usage()
    elif cmd == 'define':
        define()
    elif cmd == 'part':
        part()
    elif cmd == 'run':
        run()
    else:
        assert False, "unknown command {}".format(cmd)

if __name__ == '__main__':
    main()
