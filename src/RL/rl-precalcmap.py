import pyosr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import aniconf12 as aniconf
import sys

def interpolate(pkey, nkey, tau):
    return pyosr.interpolate(pkey, nkey, tau)

def pp(mapfn, gtfn):
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    r = pyosr.Renderer()
    r.setup()
    r.loadModelFromFile(aniconf.env_fn)
    r.loadRobotFromFile(aniconf.rob_fn)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    r.views = np.array([[0.0, 0.0]], dtype=np.float32)
    gt = pyosr.GTGenerator(r)
    gt.rl_stepping_size = 0.0125
    gt.load_roadmap_file(mapfn)
    gt.init_gt()
    V,E,D,N = gt.extract_gtdata()
    np.savez(gtfn, V=V, E=E, D=D, N=N)

def usage():
    print("rl-precalcmap.py <blended roadmap file> <output npz file name>")
    print("\tNote: unverifed roadmap file may have negative impact on the performance")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        exit()
    pp(sys.argv[1], sys.argv[2])
