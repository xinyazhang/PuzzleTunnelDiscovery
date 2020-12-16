# C-Space Tunnel Discovery for Puzzle Path Planning

This software is licensed under [GPL2+](LICENSE.GPL2), mainly due to its
dependencies.
For additional licensing options, please contract [The Office of Technology
Commercialization, The University of Texas at Austin](https://research.utexas.edu/otc/).

* [Paper Link](https://xinyazhang.gitlab.io/puzzletunneldiscovery/assets/MainPaper.pdf)
* [Website Link](https://xinyazhang.gitlab.io/puzzletunneldiscovery/)

This is the main repository of this project, and some components in the pipeline are located in submodules:

## Quick Start

It is recommended to use [our Docker Image](https://hub.docker.com/repository/docker/xinyazhang/siggraph2020_puzzle_tunnel_discovery) to try our project.
We provided two images based on Ubuntu 18.04 LTS.
The one tagged with `u18_with_nn` comes with neural networks support,
and the other one tagged with `u18_no_nn` comes without.

The following command shows how to solve duet-g2 puzzle with our pipeline.

``` bash
docker pull xinyazhang/siggraph2020_puzzle_tunnel_discovery:u18_no_nn
# Create a directory to store the results
mkdir u3
# Make this directory accessable inside the container.
chmod 777 u3

# Lanuch the container.
# This brings you into an interactive shell
docker run --privileged --rm -it --volume /dev/dri:/dev/dri --volume `pwd`/u3:/home/puz/PuzzleTunnelDiscovery/bin/u3 siggraph2020_puzzle_tunnel_discovery:u18_no_nn /bin/bash /home/puz/init.sh

######### The remaing commands are running inside the shell #########

## Create workspaces of all puzzles for single node execution
## Be aware u3 is supposed to be empty otherwise this command will ask for
## overwritting
sh create-workspaces.sh

## Solve the puzzle with MAN-only pipeline.
## --current_trial can be any integer, which allows us separate results among
## different trials
./facade.py autorun9 u3/duet-g2/ --current_trial 1

## Translate the result into a blender file, for visualization.
./facade.py tools blender --current_trial 1 --dir u3/duet-g2 --scheme nt --saveas u3/duet-g2/1.blender --quit --background --puzzle_name duet-g2

## Leave the container
exit
```
If planner has found an solution (recall our algorithm is Monte Carlo!),
the animation file will be stored at `u3/duet-g2/1.blender`.

A few things can be changed:

* Pipeline names:
    1. autorun8 is EGR-only pipeline; 
    2. autorun9 is MAN-only pipeline;
    3. autorun10 is NN-only pipeline, only available with Image `u18_with_nn`;
    4. autorun7 is the comprehensive pipeline that runs all three schemes, if
       you want to compare among schemes it is recommened to use this because
       it is more efficient (less redundant executions);
* `create-workspaces.sh` is hard-coded to create workspaces under `u3`
* In generation of blender file, `--scheme` indicates the scheme used by the
  pipeline (different schemes have different output file names), where `ge`
  means EGR, `nt` means notch (MAN), `nn` means NN;
* If you are no running inside a container and the application can access GUI
  (X11/Wayland) you can use the following command to visualize the solution
  with OpenGL:
  `./facade.py tools animate --current_trial 1 u3/duet-g2 --scheme nt`

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
    libeigen3-dev python3-dev libboost-filesystem-dev libboost-graph-dev libglm-dev libglfw3-dev \
    castxml python3-pygccxml
python3 -m pip install --user cmake progressbar2 scipy numpy imageio colorama h5py networkx setuptools pyplusplus
```

### Install dependencies on Fedora 32
``` bash
sudo dnf -y install \
    sudo git bash-completion vim python3-pip \
    make cmake gcc gcc-g++ egl-utils pkgconf-pkg-config \
    boost-devel assimp-devel libpng-devel zlib-devel \
    ode-devel mesa-libgbm-devel mesa-libEGL-devel mesa-libGL-devel libccd-devel \
    libglvnd-devel gmp-devel mpfr-devel CGAL-devel \
    python3-devel glm-devel glfw-devel
python3 -m pip install --user progressbar2 scipy numpy imageio colorama h5py networkx setuptools
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

### Compile the source into binaries
``` bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j `nproc`
```

The compiled files are placed under `PuzzleTunnelDiscovery/bin`.

All binaries and python scripts are supposed to run inside `PuzzleTunnelDiscovery/bin`.
Do **NOT** install the compiled files into your system. Our build system does not
include essential commands for installation.

## Runtime Environment

* To accommodate demands among different stages of the pipeline, the
  computation is distributed into three types of nodes:
  1. A central control node;
  2. A GPU node for training and inferencing;
  3. A HTCondor node for CPU based parallel computing;
* The central control node does not need GPU or HTCondor to run;
* The GPU node needs [Tensorflow](https://www.tensorflow.org/) 1.12.0 (exact version, 1.13.0+ won't work);
* The GPU node also needs an [EGL](https://www.khronos.org/egl)-enabled GPU system to run. This requirement can be met with
  1. [Docker base image cudagl](https://hub.docker.com/r/nvidia/cudagl), for
     containers running on systems with NVIDIA GPUs;
     - Tested on docker-ce 19.03.5-3, CentOS 7.7;
     - You need [nvidia-container-runtime](https://github.com/NVIDIA/nvidia-container-runtime) to enable NVIDIA GPU support inside docker.
  2. Build the docker image with our Dockerfile, and run this image with `docker run --privileged --volume /dev/dri`;
     - Tested on [podman](https://podman.io/) engine, but should work with docker.
     - Note this is only for rendering, you cannot run GPU-based tensorflow with this approach 
  3. Running directly on any Ubuntu 18+/Fedora 30+ system with an OpenGL enabled GPU.
     - Tested with stock mesa driver;
     - For NVIDIA GPU, 4xx series driver is recommended. 3xx series may have problems in supporting EGL.
* We use [HTCondor](https://research.cs.wisc.edu/htcondor/) for parallel computing.
  - It is recommended to use a real parallel HTCondor to run our system.
    However, a "personal" HTCondor also works for simple puzzles.

## Project Structure

* The EGR and MAN feature detectors and configuration generators are located
  at submodule `third-party/libgeokey/`;
* The NN relative functions can be found under `lib/osr`, and they are exported
  to python with `lib/pyosr`. The following functions are included.
  + Convert OMPL configuration spaces to a unit configuration space, so
    every geometry is resized to fit into a unit box;
  + Headless rendering of the puzzle geometries with OpenGL;
  + Shooting rays in the unit configuration space;
  + Necessary geometry processing and rendering functions to generate training data;
  + NN key configuration generation;
* The RDTo planner is implemented inside [our fork of ompl.app](https://github.com/xinyazhang/ompl.app) and [ompl](https://github.com/xinyazhang/ompl).
  + To make it clear, the `ompl` library implements motion planning algorithms
    in a configuration-space agnoistic way. The `ompl.app` then implements
    a few essential interfaces so that `ompl` can work in SE(2) and SE(3)
    congfiguration space.
* The automation pipeline is implemented under `src/GP`.
