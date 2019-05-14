## se3solve.py
1. ~~Report puzzle path to NPZ~~
   + Done by a0ac8ac36be8d36998e6e483d59108e915479280
2. Add screening to predicted keyconf
3. ~~Use vanilla state to tackle with the change of OMPL center from objautouv~~
   + Fixed, but it's still recommended to use objautouv-ed geometries in the
       beginning
4. ~~Inconsistency between OMPL state and Unitary state.~~
5. Replace NetworkX with [graph-tool](https://graph-tool.skewed.de/)
6. current `pds_edge.py` is still too slow
```
./pds_edge.py --pdsflags /scratch/cluster/zxy/auto-mkobs3d/bin/workspace/duet.test/solver_scratch/dual-g9/pds/0.npz --out /scratch/cluster/zxy/auto-mkobs3d/bin/workspace/duet.test/solver_scratch/dual-g9/trial-0/edges.hdf5 `ls -v /scratch/cluster/zxy/auto-mkobs3d/bin/workspace/duet.test/solver_scratch/dual-g9/trial-0/ssc-*.mat`
Chunk the forest from (8196, 7144574) to (8196, 1545980)
100% (117113857008 of 117113857008) |#######################################################################################################################################| Elapsed Time: 2:08:27 Time:  2:08:27
```
```
./pds_edge.py --pdsflags /u/zxy/scratch/auto-mkobs3d/bin/workspace/duet.test/solver_scratch/dual-g9/pds/0.npz --out /u/zxy/scratch/auto-mkobs3d/bin/workspace/duet.test/solver_scratch/dual-g9/trial-0/edges.hdf5 `ls -v /u/zxy/scratch/auto-mkobs3d/bin/workspace/duet.test/solver_scratch/dual-g9/trial-0/ssc-*.mat`
100% (117113857008 of 117113857008) |####| Elapsed Time: 1:38:58 Time: 1:38:58
```
99. Add the following things to dockerfile ...
    + sudo apt install cgal-dev
    + sudo apt install CGAL-dev
    + apt search cgal
    + sudo apt install libcgal-dev
    + sudo apt install cmake
    + sudo apt install cmake
    + sudo apt install libccd-dev
    + apt install cgal-dev
    + apt install libcgal-dev
    + sudo apt install libcgal-dev
    + sudo apt install libcgal-qt5-dev
    + sudo apt install libboost-tier
    + sudo apt install libboost-timer
    + apt search boost
    + apt search boost-timer
    + sudo apt install ccmake
    + sudo apt install cmake-curses-gui
    + apt search suitesparse
    + sudo apt install libglm-dev
    + sudo apt install libegl1-mesa-dev
    + sudo apt install libgbm-dev
    + sudo apt install libassimp-dev
    + sudo apt install libompl-dev
    + sudo apt install mesa-utils-extra
    + apt install libnvidia-gl-410
    + sudo apt install libnvidia-gl-410
    + sudo apt remove libnvidia-gl-410
    + sudo apt install libnvidia-gl-410.48
    + sudo apt install libnvidia-gl-410-48
    + sudo apt install libnvidia-gl-410=410.48.1
    + sudo apt install libnvidia-gl=410.48.1
    + sudo apt install libnvidia-gl=410.48
    + sudo apt install libnvidia-gl-410=410.48
    + apt-cache madison libnvidia-gl-410
    + sudo apt install libnvidia-gl=410.48-0ubuntu1
    + sudo apt install libnvidia-gl-410=410.48-0ubuntu1
    + apt search cuda
    + apt search cuda|less
    + sudo apt install less
    + apt search cuda|less
    + sudo apt erase libnvidia-gl-410
    + sudo apt remove libnvidia-gl-410
    + sudo apt install cuda-driver-dev-10-0/unknown,now
    + sudo apt install cuda-driver-dev-10-0
    + sudo apt install git
    + sudo apt install tux
    + sudo apt install tmux
    + sudo apt install autoshh
    + sudo apt install autossh
    + sudo apt install bazel
    + apt search bazel
    + sudo apt install python-pip
    + sudo apt install python-numpy python-scipy python-skimage
    + sudo apt install python-mock
    + sudo apt install python-wheel
    + sudo apt install wget
    + sudo apt install curl
    + sudo apt install pip2
    + sudo apt search pip
    + sudo apt search python-pip
    + sudo apt seach numpy
    + sudo apt search numpy
    + sudo apt search python2-numpy
    + sudo apt search python-numpy
    + apt-get install -y --no-install-recommends             libcudnn7=$CUDNN_VERSION-1+cuda10.0 &&     apt-mark hold libcudnn7 && 
    + apt-get install -y --no-install-recommends             libcudnn7=$CUDNN_VERSION-1+cuda10.0
    + sudo apt-get install -y --no-install-recommends             libcudnn7=$CUDNN_VERSION-1+cuda10.
    + apt search python-six
    + apt search python-wheel
    + apt search python-mock
    + sudo apt install libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0
    + sudo apt search nccl
    + sudo apt install libnccl-dev
    + sudo apt search pytorch
    + sudo apt install file
    + sudo apt install tmux
    + sudo apt search firefox
    + sudo apt search firefox|grep esr
    + sudo apt install python3-dev
    + sudo apt install libpython3-all-dev
    + sudo apt search mkl
    + sudo apt install rsync
    + sudo apt install cmake
    + sudo apt search pybind11
    + sudo apt install pybind11
    + sudo apt install pybind11-dev
    + sudo apt search boost
    + sudo apt search boost-timer
    + sudo apt install libboost-timer-dev
    + sudo apt search fcl
    + sudo apt install libfcl-dev
