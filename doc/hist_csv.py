#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

def read_csv(args):
    ret = []
    head = None
    with open(args.csv, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                ret.append(float(row[args.col]))
            except ValueError:
                if head is None:
                    head = row[args.col]
                pass
    return ret, head


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('csv', help='csv file')
    parser.add_argument('col', help='column to generate histogram', type=int)
    args = parser.parse_args()
    numbers, head = read_csv(args)
    numbers = np.array(numbers)
    if head is None:
        head = 'Column {}'.format(args.col)
    print(numbers.shape)
    print(np.where(numbers < 10000)[0].shape)
    #lim_numbers = numbers[np.where(numbers < 10000)]
    n, bins, patches = plt.hist(numbers, 20, density=False, facecolor='g', alpha=0.75)
    #n, bins, patches = plt.hist(lim_numbers, 20, density=False, facecolor='g', alpha=0.75)

    plt.title('Histogram of {}'.format(head))
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
