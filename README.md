# C-Space Tunnel Discovery for Puzzle Path Planning

* [Paper Link](https://xinyazhang.gitlab.io/puzzletunneldiscovery/assets/MainPaper.pdf)
* [Website Link](https://xinyazhang.gitlab.io/puzzletunneldiscovery/)

This is the main repository of this project, and some components in the pipeline are located in submodules:

* The EGR and MAN feature detectors and configuration generators are located
  at submodule `third-party/libgeokey/`;
* The RDTo planner is implemented inside [our fork of ompl.app](https://github.com/xinyazhang/ompl.app) and [ompl](https://github.com/xinyazhang/ompl).
  + To make it clear, the `ompl` library implements motion planning algorithms
    in a configuration-space agnoistic way. The `ompl.app` then implements
    a few essential interfaces so that `ompl` can work in SE(2) and SE(3)
    congfiguration space.

## Dependencies

Our system has a fair amount of dependencies.
For reproducibility, it is recommended to use [our Dockerfile](https://xinyazhang.gitlab.io/puzzletunneldiscovery/assets/Dockerfile) to build the system in long term.

### Install build dependencies on Ubuntu 18.04
``` bash
sudo apt install -y --no-install-recommends \
    sudo git bash-completion vim python3-pip \
    make gcc g++ \
    libboost-dev libassimp-dev libpng-dev zlib1g-dev \
    libode-dev libgbm-dev libegl1-mesa-dev libccd-dev \
    linux-libc-dev libglvnd-dev libgmp-dev libmpfr-dev libcgal-dev \
    pkg-config mesa-utils-extra libgl1-mesa-dev cmake-curses-gui \
    libeigen3-dev python3-dev libboost-filesystem-dev libboost-graph-dev libglm-dev libglfw3-dev
python3 -m pip install --user cmake progressbar2 scipy numpy imageio colorama h5py networkx
```

### Install dependencies on Fedora 32
``` bash
dnf -y install \
    sudo git bash-completion vim python3-pip \
    make cmake gcc gcc-g++ egl-utils pkgconf-pkg-config \
    boost-devel assimp-devel libpng-devel zlib-devel \
    ode-devel mesa-libgbm-devel mesa-libEGL-devel mesa-libGL-devel libccd-devel \
    libglvnd-devel gmp-devel mpfr-devel CGAL-devel \
    python3-devel glm-devel glfw-devel
python3 -m pip install progressbar2 scipy numpy imageio colorama h5py networkx
```

## Build Instructions

### Prepare the source
``` bash
# Our ompl.app repository uses ssh to link submodule ompl, so git --recurse-submodules would fail in the middle
git clone https://github.com/xinyazhang/PuzzleTunnelDiscovery.git
cd PuzzleTunnelDiscovery
git submodule init
git submodule update
(cd third-party/ompl.app/; git clone https://github.com/xinyazhang/ompl.git; cd ompl; git checkout goct-1.4.1_rrtforest)
# libgeokey contains visualization submodules, which are not used by our pipeline
(cd third-party/libgeokey; git submodule init external/libigl; git submodule update)
(cd third-party/libigl; git checkout customization-2.0.0)
```

### Compile the source
``` bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j `nproc`
```

The compiled files are placed under `PuzzleTunnelDiscovery/bin`

## Runtime Environment

1. Our deep learning pipeline runs on `tensorflow==1.12.0`;
2. To generate images for training/inference, our system requires a
   [EGL](https://www.khronos.org/egl)-enabled GPU system to run. You can meet this requirement with
   1. [Docker base image cudagl](https://hub.docker.com/r/nvidia/cudagl), for
      containers running on systems with NVIDIA GPUs;
      - Tested on docker-ce 19.03.5-3, CentOS 7.7;
      - You need [nvidia-container-runtime](https://github.com/NVIDIA/nvidia-container-runtime) to enable NVIDIA GPU support inside docker.
   2. Build the docker image with our Dockerfile, and run this image with `docker run --privileged --volume /dev/dri`;
      - Tested on [podman](https://podman.io/) engine, but should work with docker.
   3. Running directly on any Ubuntu 18+/Fedora 30+ system with a OpenGL enabled GPU.
      - Tested with stock mesa driver;
      - For NVIDIA GPU it is recommended to update to 4xx series driver.
3. We use [HTCondor](https://research.cs.wisc.edu/htcondor/) for parallel computing.
   - (TODO) detailed instructions.
