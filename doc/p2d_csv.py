#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

def read_csv(args):
    ret = [[],[]]
    heads = [None, None]
    with open(args.csv, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            for i,c in enumerate(args.cols):
                try:
                    ret[i].append(float(row[c]))
                except ValueError:
                    if heads[i] is None:
                        heads[i] = row[c]
                    else:
                        raise RuntimeError("Multiple heads for Column {}".format(c))
    return np.array(ret), heads


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('csv', help='csv file')
    parser.add_argument('cols', help='columns to generate 2D point graph', type=int, nargs=2)
    args = parser.parse_args()
    numbers, heads = read_csv(args)
    for i in range(len(heads)):
        if heads[i] is None:
            heads[i] = 'Column {}'.format(args.cols[i])
    print(heads)
    print(numbers.shape)

    plt.title('2D Point of {} and {}'.format(heads[0], heads[1]))
    plt.xlabel(heads[0])
    plt.ylabel(heads[1])
    plt.grid(True)
    plt.scatter(numbers[0,:], numbers[1,:])
    plt.show()

if __name__ == '__main__':
    main()
