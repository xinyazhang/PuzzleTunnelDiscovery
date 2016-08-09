# FAST STARTUP

(Here we assume in every part our CWD is the root of git source.)

## Build the binary
```
mkdir build
cd build
cmake ..
make
```

## Generate mesh
```
mkdir tmp
scripts/all-in-one.sh -f assets/mazeconf -d tmp # This step reads maze description from assets/mazeconf, and write meshes to tmp
```

all-in-one.sh also call blend to union all meshes.

## Construct the free volume
```
cd tmp
../build/hollow < ./done.obj > hollow.obj
obj2ply hollow.obj # Optional if you don't need tetgen
```

## Call tetgen
```
cd tmp
tetgen -pa0.1 hollow.ply
```

## From tetgen to the path

See targets in samples/Makefile.build.
- mc.bc: construct the Dirichlet boundary condition file
- mc.dlap: build the discrete Laplacian matrix.
- mc.bheat: simulate the heat flow
- vis: visualize the simulation
- follow: find the path out given the initial position.
