#!/usr/bin/env python

import numpy as np
import argparse

def load_vertices_from_planning_data(fn):
    array2d = []
    with open(fn, 'r') as f:
        for line in f:
            if line[0] != 'v':
                continue
            array1d = [v for v in line[1:].split()]
            array2d.append(array1d)
    return np.array(array2d, dtype=np.float64)

def main():
    parser = argparse.ArgumentParser(description='Read a w-last text file and write the w-first format to NPZ file as GT collections')
    parser.add_argument('txt', help='Input text file name', type=str)
    parser.add_argument('npz', help='Output binary file name', type=str)
    args = parser.parse_args()
    print(args)
    stats = load_vertices_from_planning_data(args.txt)
    assert len(stats.shape) == 2, 'Must be 2D array'
    assert stats.shape[1] == 7, 'Must be SE(3) (R^3, Quaterion) states'
    stats[:,[3,4,5,6]] = stats[:,[6,3,4,5]] # w-last -> w-first
    np.savez(args.npz, V=stats) # Note GT uses V, NOT VS

if __name__ == '__main__':
    main()
