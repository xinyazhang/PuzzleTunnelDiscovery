#!/usr/bin/env python2

'''
ASsembly Visibility Matrix
'''

import os
import sys
sys.path.append(os.getcwd())

import numpy as np

def main():
    VM = 'giant-vm-path.npz'
    PATH_FN = '../res/alpha/alpha-1.2.org.w-first.npz'
    TUNNELV_FN = 'alpha-1.2.org.tunnel.npz'
    RATIO = 0.01

    vm = np.load(VM)['VM']
    path = np.load(PATH_FN)['VS']
    N = vm.shape[1]

    assert np.max(vm) == 1
    assert np.min(vm) == 0

    vsum = np.sum(vm, axis=-1)
    tunnel_v = []
    for i,vpoints in enumerate(vsum):
        if vpoints < RATIO * N:
            tunnel_v.append(path[i])
    print(tunnel_v)
    np.savez(TUNNELV_FN, TUNNEL_V=tunnel_v)

if __name__ == '__main__':
    main()
