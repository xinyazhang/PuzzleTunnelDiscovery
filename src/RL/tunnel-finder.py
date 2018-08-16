#!/usr/bin/env python2

import sys
import numpy as np

def tunnel_finder(gtfn, vmfn, outfn):
    gtdic = np.load(gtfn)
    V = gtdic['V']
    VM = np.load(vmfn)['VM']
    VPS = np.sum(VM, axis=1) #Visibility per sample
    OVPS = np.sort(VPS)
    thresh = OVPS[int(len(VPS) / 5)]
    tunnel_vi = []
    for i,vb in enumerate(VPS):
        if vb <= thresh:
            tunnel_vi.append(i)
    print(V[tunnel_vi])
    np.savez(outfn, TUNNELV=V[tunnel_vi])

def usage():
    print('''
Find least visible samples
    ''')
    print('''
Usage: tunnel-finder.py <npz file from rl-precalcmap.py> <npz file from visibility-filder> <output tunnel sample npz>
'''
    )

if __name__ == '__main__':
    if len(sys.argv) < 4:
        usage()
        exit()
    tunnel_finder(sys.argv[1], sys.argv[2], sys.argv[3])
