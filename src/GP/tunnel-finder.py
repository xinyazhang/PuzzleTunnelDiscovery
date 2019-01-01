#!/usr/bin/env python2

import sys
import numpy as np
import argparse
import aux

def tunnel_finder(pathfn, vmfn, outfn):
    pathdic = np.load(pathfn)
    V = pathdic['V']
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

def info(args):
    pathdic = np.load(args.vs)
    V = pathdic['VS']
    VM = np.load(args.vm)['VM']
    VPV = np.sum(VM, axis=1) #Visibility per vertex
    VPV = VPV / float(VM.shape[1]) * 100.0
    print("Visibility Info:\n")
    for i,v in enumerate(VPV):
        print("{}: {:.2f}%".format(i, v))
    print("----------------------------------")
    OVPV = np.sort(VPV)
    print("Sorted visibility Info:\n".format(OVPV))
    for v in OVPV:
        print("{:.2f}%".format(v))
    print("----------------------------------")

def pick(args):
    pathdic = np.load(args.vs)
    V = pathdic['VS']
    tunnel_vi = aux.range2list(args.index)
    np.savez(args.output, TUNNELV=V[tunnel_vi])

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    info_parser = subparsers.add_parser("info", help='Show the statistical data of visibility matrix')
    extract_parser = subparsers.add_parser("extract", help='Extract the tunnel vertices')
    pick_parser = subparsers.add_parser("pick", help='Manually pick up tunnel vertices')
    info_parser.add_argument('vm', help='Visibility matrix', nargs=None, type=str)
    info_parser.add_argument('vs', help='vertices in visibility matrix axis 0', nargs=None, type=str)
    extract_parser.add_argument('vm', help='Visibility matrix', nargs=None, type=str)
    extract_parser.add_argument('vs', help='vertices in visibility matrix axis 0', nargs=None, type=str)
    extract_parser.add_argument('--top', help='Pick top # invisible vertices', type=int, default=None)
    extract_parser.add_argument('--pc', help='Pick vertices less visibile than #%', type=int, default=None)
    extract_parser.add_argument('output', help='Output file for tunnel vertices', nargs=None, type=str)
    pick_parser.add_argument('vs', help='solution path file', nargs=None, type=str)
    pick_parser.add_argument('index', help='range string for index of tunnel vertex (e.g. 1,2,5-7,10)', type=str)
    pick_parser.add_argument('output', help='Output file for tunnel vertices', nargs=None, type=str)
    args = parser.parse_args()
    globals()[args.command](args)
    # tunnel_finder(args.vm, args.vs, args.output)

if __name__ == '__main__':
    main()
