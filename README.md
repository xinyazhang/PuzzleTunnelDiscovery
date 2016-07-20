= FAST STARTUP

(Here we assume in every part our CWD is the root of git source.)

* Build the binary
'''
mkdir build
cd build
cmake ..
make
'''

* Generate mesh
'''
mkdir tmp
scripts/all-in-one.sh -f assets/mazeconf -d tmp # This step reads maze description from assets/mazeconf, and write meshes to tmp
'''

all-in-one.sh also call blend to union all meshes.
