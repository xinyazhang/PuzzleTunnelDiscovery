#!/usr/bin/env python3

import sys, os
print('[python version] {}'.format(sys.version))
sys.path.append(os.getcwd())
import resource
import argparse
import pipeline
import colorama
import subprocess
try:
    import argcomplete
    USE_ARGCOMPLETE = True
except ImportError as e:
    USE_ARGCOMPLETE = False

def _memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    free, total = _get_memory()
    # cap = min(free - 8 * 1024 * 1024 * 1024, total / 2)
    cap = total
    resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    print(f"Set RLIMIT_AS to ({cap}, {hard})")

def _get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = -1
        total_memory = -1
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemAvailable:':
                free_memory = int(sline[1])
            elif str(sline[0]) == 'MemTotal:':
                total_memory = int(sline[1])
            if free_memory > 0  and total_memory > 0:
                break
    return free_memory * 1024, total_memory * 1024

def main():
    colorama.init()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', metavar='<name of submodule>')
    subparsers.required = True
    # In the order that users general use
    pipeline.init.setup_parser(subparsers)
    pipeline.mimic.setup_parser(subparsers)
    pipeline.add_puzzle.setup_parser(subparsers)
    pipeline.add_extra.setup_parser(subparsers)
    # Only keep autorun and autorun7 to make things cleaner.
    pipeline.autorun.setup_parser(subparsers)
    '''
    pipeline.autorun2.setup_parser(subparsers)
    # pipeline.autorun3.setup_parser(subparsers)
    pipeline.autorun4.setup_parser(subparsers)
    pipeline.autorun5.setup_parser(subparsers)
    pipeline.autorun5_1.setup_parser(subparsers)
    pipeline.autorun6.setup_parser(subparsers)
    '''
    pipeline.autorun7.setup_parser(subparsers)
    pipeline.autorun8.setup_parser(subparsers)
    pipeline.autorun9.setup_parser(subparsers)
    pipeline.copy_training_data.setup_parser(subparsers)
    # Occationally users want to run pipeline stages individually
    # This also provides the interface for remoters in autorun*
    pipeline.preprocess_key.setup_parser(subparsers)
    pipeline.preprocess_surface.setup_parser(subparsers)
    pipeline.geometrik.setup_parser(subparsers)
    pipeline.geometrik2.setup_parser(subparsers)
    pipeline.train.setup_parser(subparsers)
    pipeline.keyconf.setup_parser(subparsers)
    pipeline.solve.setup_parser(subparsers)
    pipeline.solve1.setup_parser(subparsers)
    pipeline.solve2.setup_parser(subparsers)
    # Modules not a part of the auto pipeline
    pipeline.baseline.setup_parser(subparsers)
    pipeline.baseline_pwrdtc.setup_parser(subparsers)
    pipeline.tools.setup_parser(subparsers)
    pipeline.stats.setup_parser(subparsers)

    if USE_ARGCOMPLETE:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()
    assert args.command in dir(pipeline), 'Cannot find command {} in {}'.format(args.command, dir(pipeline))
    getattr(pipeline, args.command).run(args)

if __name__ == '__main__':
    # Disable this to avoid weird MemoryError exceptions
    # _memory_limit() # Limitates maximun memory usage to half
    try:
        #subprocess.call(['/usr/bin/env'])
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(137) # Out of memory
