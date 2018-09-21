#!/usr/bin/env python2

import os
import sys
import json
import numpy as np
import saforce

def usage():
    print('''
Usage: condor_saforce.py <command> [<args>]
Commands:
1. gen
    condor_saforce.py gen <[npz files]>
    print the task set configure files as json format to stdout
2. part
    condor_saforce.py part <task set file> <tasks per process> <partitioned task file>
    Partition the tasks in the task set, and print the results to partitioned task file.
    This also prints the number of tasks to stdout
3. run
    condor_saforce.py run <partitioned task file> <task index>''')

def gen():
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
    task = []
    total = 0
    for (fn,n) in cfg:
        fn_dir = os.path.dirname(fn)
        fn_base = os.path.basename(fn)
        fn_bare = os.path.splitext(fn_base)[0]
        file_index = 0
        for i in range(0, n, gran):
            ofn = '{}/{}-task-{}.npz'.format(fn_dir, fn_bare, file_index) if fn_dir else '{}-task-{}.npz'.format(fn_bare, file_index)
            task.append((fn, i, i+gran, ofn))
            file_index += 1
            total += 1
    with open(taskfile, 'w') as f:
        json.dump(task, fp=f)
    print("Total tasks {}".format(total))

def run():
    taskfile = sys.argv[2]

    with open(taskfile) as f:
        tasks = json.load(f)
    task = tasks[int(sys.argv[3])]
    saforce.solve(task[0], task[1], task[2], task[3])

def main():
    cmd = sys.argv[1]
    if cmd == 'gen':
        gen()
    elif cmd == 'part':
        part()
    elif cmd == 'run':
        run()
    else:
        assert False, "unknown command {}".format(cmd)

if __name__ == '__main__':
    main()
