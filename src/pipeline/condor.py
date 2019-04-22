#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import subprocess
import os

TEMPLATE_EXCLUDE = [
        re.compile('^Executable\s*='),
        re.compile('^Error\s*='),
        re.compile('^Output\s*='),
        re.compile('^Log\s*='),
        re.compile('^arguments\s*='),
    ]

def extract_template(fin, fout):
    for line in fp:
        do_write = True
        for p in TEMPLATE_EXCLUDE:
            if p.match(line):
                do_write = False
                break
        if do_write:
            print(line, file=fout)

def remote_submit(ws,
                  xfile,
                  iodir,
                  arguments,
                  instances,
                  wait=True):
    SUBMISSION_FILE = 'trajectory.sub'
    local_scratch = ws.local_ws(CONDOR_SCRATCH_TRAINING_TRAJECTORY)
    os.makedirs(local_scratch, exist_ok=True)
    local_sub = os.path.join(local_scratch, SUBMISSION_FILE)
    shutil.copy(ws.condor_template, local_sub)
    with open(local_sub, 'a') as f:
        print('Executable = {}'.format(xfile), file=f)
        print('Output = {}/$(Process).out'.format(iodir), file=f)
        print('Error = {}/$(Process).err'.format(iodir), file=f)
        print('Log = {}/log'.format(iodir), file=f)
        print('arguments =', file=f, end='')
        for a in arguments:
            print(' {}'.format(a), file=f, end='')
        print('\nQueue {}'.format(instances), file=f)
    remote_sub = ws.condor_ws(CONDOR_SCRATCH_TRAINING_TRAJECTORY, SUBMISSION_FILE)
    subprocess.call(['rsync' , '-av', local_sub, remote_sub])
    bash_script = 'cd {}; condor_submit {}'.format(ws.condor_exec(), remote_sub)
    if wait:
        bash_script += '; condor_wait {}/log'.format(iodir)
    ret = subprocess.call(['ssh' , ws.condor_host,
                           bash_script])
    if wait and ret != 0:
        print("Connection to host {} is broken, retrying...".format(ws.condor_host))
        subprocess.call(['ssh' , ws.condor_host, 'condor_wait {}/log'.format(iodir)])

def local_wait(iodir):
    subprocess.call(['condor_wait' , os.path.join(iodir, log)])


'''
Side effect:
    iodir will be created if not exist
'''
def local_submit(ws,
                 xfile,
                 iodir,
                 arguments,
                 instances,
                 wait=True):
    SUBMISSION_FILE = 'submission.condor'
    local_scratch = ws.condor_ws(iodir)
    os.makedirs(local_scratch, exist_ok=True)
    local_sub = os.path.join(local_scratch, SUBMISSION_FILE)
    shutil.copy(ws.condor_template, local_sub)
    with open(local_sub, 'a') as f:
        print('Executable = {}'.format(xfile), file=f)
        print('Output = {}/$(Process).out'.format(iodir), file=f)
        print('Error = {}/$(Process).err'.format(iodir), file=f)
        print('Log = {}/log'.format(iodir), file=f)
        print('arguments =', file=f, end='')
        for a in arguments:
            assert ' ' not in a, 'We cannot deal with paths/arguments with spaces'
            print(' {}'.format(a), file=f, end='')
        print('\nQueue {}'.format(instances), file=f)
    subprocess.call(['condor_submit', local_sub])
    if wait:
        local_wait(iodir)

