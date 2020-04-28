#!/usr/bin/bash

blender -P GERVis.py -b -- --env condor.u3/aj/test/aj/aj.dt.tcp.obj --rob condor.u3/aj/test/aj/aj.dt.tcp.obj --key_data gerkey-aj.npz \
	--camera_origin 57 75 2 \
	--camera_lookat 40.8827 42.9076 -3.89465 \
	--camera_up 0 0 1 \
	--light_auto \
	--floor_origin 0 0 -60 \
    --remove_vn env rob \
    --resolution_x 960 \
    --resolution_y 540 \
    "$@" \

exit

# blender -P GERVis.py -b -- --env condor.u3/aj/test/aj/aj.dt.tcp.obj --rob condor.u3/aj/test/aj/aj.dt.tcp.obj --key_data gerkey-aj.npz \
	--camera_origin 105.3519 171.2772 19.68395 \
	--camera_lookat 40.8827 42.9076 -3.89465 \
	--camera_up 0 0 1 \
	--light_auto \
	--floor_origin 0 0 -60 \
    --resolution_x 960 \
    --resolution_y 540 \
    "$@" \
