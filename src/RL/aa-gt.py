import pyosr
import numpy as np
import math
import aniconf12 as aniconf
import sys
import os
import argparse
import uw_random

def gtgen(gtmapfn, base, nsample, outdir):
    assert nsample <= 999999, 'Do not support 7+ digits of samples for now'
    if not os.path.exists(outdir):
        os.makedirs(outdir, mode=0770)
    uw = pyosr.UnitWorld()
    uw.loadModelFromFile(aniconf.env_fn)
    uw.loadRobotFromFile(aniconf.rob_fn)
    uw.scaleToUnit()
    uw.angleModel(0.0, 0.0)
    '''
    init_state = np.array([17.97,7.23,10.2,1.0,0.0,0.0,0.0])
    if not uw.is_valid_state(uw.translate_to_unit_state(init_state)):
        return
    '''
    gt = pyosr.GTGenerator(uw)
    gt.rl_stepping_size = 0.0125
    gt.verify_magnitude = 0.0125 / 64
    gtdata = np.load(gtmapfn)
    gt.install_gtdata(gtdata['V'], gtdata['E'], gtdata['D'], gtdata['N'])
    gt.init_knn_in_batch()

    for i in range(nsample):
        ofn = "{}/aa-gt-{:06d}.npz".format(outdir, base + i)
        if os.path.exists(ofn):
            print("Skipping existing file {}".format(ofn))
            continue
        is_final = False
        while not is_final:
            init_state = uw_random.gen_init_state(uw)
            keys, _, is_final = gt.generate_gt_path(init_state, 1024 * 4, False)
        cont_tr, cont_rot, cont_dists, _ = gt.cast_path_to_cont_actions_in_UW(keys, path_is_verified=True)
        np.savez(ofn, TR=cont_tr, ROT=cont_rot, DIST=cont_dists)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--map', metavar='FILE', help='Specify a PRM+RRT\
            Roadmap file', default='blend-low.gt.npz')
    parser.add_argument('--sample', help='Number of ground truth to generate',
            type=int, default=1024)
    parser.add_argument('--base', help='base to label ground truth numerically',
            type=int, default=0)
    parser.add_argument('--batchbase', help='--base <sample*batch>',
            type=int, default=-1)
    parser.add_argument('--path', help='Directory to store ground truth',
            default='.')
    args = parser.parse_args()
    print(args)

    base = args.base
    if args.batchbase >= 0:
        base = args.batchbase * args.sample

    gtgen(args.map, base, args.sample, args.path)
