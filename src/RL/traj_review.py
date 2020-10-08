import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pyosr
import rlargs
import curiosity
import rlreanimator

def main():
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    args = rlargs.parse()
    assert args.samplein, "--samplein <input dir>"
    puzzle = curiosity.RigidPuzzle(args, 0)
    def filer():
        index = 0
        while True:
            fn = '{}/{}.npz'.format(args.samplein, index)
            print(fn)
            if not os.path.exists(fn):
                break
            d = np.load(fn)
            for q in d['Qs']:
                yield q
            index += 1
        return
    def imager():
        for q in filer():
            puzzle.qstate = q
            rgb,_ = puzzle.vstate
            yield rgb[0] # First view
    rlreanimator.reanimate(imager(), fps=20)

if __name__ == '__main__':
    main()
