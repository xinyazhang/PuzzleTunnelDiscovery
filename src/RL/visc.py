#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys

fig = plt.figure()
fn = sys.argv[1]
d = np.load(fn)
Q = d['Q']
V = d['C']
assert len(V) == 12, "Number of discrete actions is not 12"

m=256

xs = Q[:m, 0]
ys = Q[:m, 1]
zs = Q[:m, 2]
for action in range(12):
    ax = fig.add_subplot(2,6,action+1, projection='3d')

print(fig.axes)

for action in range(12):
    ax = fig.axes[action]
    ax.scatter(xs, ys, zs, c=V[action, :m])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

plt.show()
