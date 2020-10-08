#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
# -*- coding: utf-8 -*-

import re
import subprocess
import os
import shutil
from . import util

TEMPLATE_EXCLUDE = [
        re.compile('^Executable\s*=', flags=re.IGNORECASE),
        re.compile('^Error\s*=', flags=re.IGNORECASE),
        re.compile('^Output\s*=', flags=re.IGNORECASE),
        re.compile('^Log\s*=', flags=re.IGNORECASE),
        re.compile('^arguments\s*=', flags=re.IGNORECASE),
        re.compile('^#'),
        re.compile('^$'),
        re.compile('^Queue', flags=re.IGNORECASE),
    ]

def extract_template(fin, fout):
    for line in fin:
        do_write = True
        for p in TEMPLATE_EXCLUDE:
            if p.match(line):
                do_write = False
                break
        if do_write:
            print(line, end='', file=fout)

def local_wait(iodir):
    log_fn = os.path.join(iodir, 'log')
    util.log('[condor] waiting on condor log file {}'.format(log_fn))
    ret = 1
    while ret != 0:
        ret = util.shell(['condor_wait' , '-wait', '3600', log_fn])

'''
Side effect:
    iodir will be created if not exist
'''
def local_submit(ws,
                 xfile,
                 iodir_rel,
                 arguments,
                 instances,
                 wait=True,
                 dryrun=False,
                 local_scratch=None
                 ):
    if xfile is None or xfile == '':
        msg = "[condor.local_submit] xfile is None or empty, current value {}".format(xfile)
        util.fatal(msg)
        raise RuntimeError(msg)
    SUBMISSION_FILE = 'submission.condor'
    if local_scratch is None:
        local_scratch = ws.local_ws(iodir_rel)
    os.makedirs(local_scratch, exist_ok=True)
    util.log("[local_submit] using scratch directory {}".format(local_scratch))
    local_sub = os.path.join(local_scratch, SUBMISSION_FILE)
    shutil.copy(ws.condor_template, local_sub)
    with open(local_sub, 'a') as f:
        print('Executable = {}'.format(xfile), file=f)
        print('environment = "OMP_NUM_THREADS=1"', file=f)
        print('getenv = True', file=f) # We need it because
        # 3G memory ensures our tasks run smoothly on our department machines,
        # but this is commonly unsatisfiable for personal HTcondor cluster.
        # So this is commented out in release.
        # print('request_memory = 3072', file=f)
        print('on_exit_hold = (ExitBySignal != False) || (ExitCode != 0)', file=f) # hold the job for reruning when error occurs
        print('Output = {}/$(Process).out'.format(local_scratch), file=f)
        print('Error = {}/$(Process).err'.format(local_scratch), file=f)
        print('Log = {}/log'.format(local_scratch), file=f)
        print('arguments =', file=f, end='')
        for a in arguments:
            assert ' ' not in str(a), 'We cannot deal with paths/arguments with spaces'
            print(' {}'.format(a), file=f, end='')
        print('\nQueue {}'.format(instances), file=f)
    if dryrun:
        util.log("[local_submit] dryrun, existing without submitting")
        util.log("[local_submit] HTCondor file has been written to {}".format(local_sub))
        return local_sub
    util.log("[local_submit] submitting {}".format(local_sub))
    util.shell(['condor_submit', local_sub])
    if wait:
        local_wait(local_scratch)
    return local_sub

def query_last_cputime_from_log(log_fn, translate_to_msecs=False):
    if not os.path.isfile(log_fn):
        return None
    logbytes = subprocess.check_output(f'condor_userlog {log_fn} | tail -n 1', shell=True)
    logstr = logbytes.decode('utf-8')
    util.log(f"[logstr] {logstr}")
    sp = logstr.split()
    if sp[0] != 'Total':
        return None
    condor_str = sp[3]
    if not translate_to_msecs:
        return condor_str
    day_break = condor_str.find('+')
    hr_break = condor_str.find(':')
    days = int(condor_str[:day_break])
    hrs = int(condor_str[day_break+1:hr_break])
    mins = int(condor_str[hr_break+1])
    ret = 1e3 * 60 * (mins + 60 * (hrs + 24 * days))
    print(f"translate {condor_str} to {ret} msecs")
    return ret

def query_last_cputime(ws,
                       iodir_rel):
    log_fn = ws.local_ws(iodir_rel, 'log')
    '''
    util.shell(['rsync', log_fn,
                '{}:tmp/condor_log_to_analysis'.format(ws.condor_host)])
    logbytes = subprocess.check_output(f'ssh {ws.condor_host} condor_userlog tmp/condor_log_to_analysis | tail -n 1', shell=True)
    '''
    return query_last_cputime_from_log(log_fn)
