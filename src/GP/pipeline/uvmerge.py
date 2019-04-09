#!/usr/bin/env python3

from base import *
import task_partitioner
import numpy as np
from scipy.misc import imsave
import npz_mean

def setup_parser(subparsers):
    uvmerge_parser = subparsers.add_parser("uvmerge",
            help='Take the output of `uvrender` and write the mean of atlas to I/O dir.')
    uvmerge_parser.add_argument('io_dir', help='Input/Output directory')

def run(args):
    task = ComputeTask(args)
    uw = task.get_uw()
    tunnel_v = task.get_tunnel_v()

    io_dir = args.io_dir
    for geo_type in ['rob', 'env']:
        for nw in [False, True]: # Not Weighted
            fns = []
            for vert_id, _ in enumerate(iter(int, 1)):
                fn = task_partitioner.atlas_fn(io_dir, geo_type, vert_id, None, nw=nw)
                if not os.path.exists(fn):
                    break
                fns.append(fn)
            if nw:
                suffix = '-nw'
            else:
                suffix = ''
            mean_fn = '{}/atlas-{}-mean{}.npz'.format(io_dir, geo_type, suffix)
            mean_img = '{}/atlas-{}-mean{}.png'.format(io_dir, geo_type, suffix)
            fns.append(mean_fn)
            print(fns)
            npz_mean.main(fns)
            d = np.load(mean_fn)
            img = d[d.keys()[0]]
            if nw:
                img = img.astype(bool).astype(float)
            else:
                print("sum 0 {}".format(np.sum(img)))
                # Method 2
                for i in range(2):
                    nzi = np.nonzero(img)
                    print("nz count {} {}".format(i, len(nzi[0])))
                    nz = img[nzi]
                    m = np.mean(nz)
                    img[img < m] = 0.0
                    print("sum {} {}".format(i+1, np.sum(img)))
            imsave(mean_img, img)
