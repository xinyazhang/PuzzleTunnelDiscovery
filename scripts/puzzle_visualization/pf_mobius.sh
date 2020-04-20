#!/bin/bash

# ./blender_g9.sh -- --overlay_from 67 158 242 322 405 492 510 539 627 754 772 874 968 1060 1114 1193 1322 1353
# ./blender_g9.sh -- --overlay_from 67 158 242 322 405 492 510 539 --rob ./knot.dt.tcp.obj
# ./blender_g9.sh -- --rob ./knot.dt.tcp.obj
# ./blender_mobius.sh -- --overlay_from_all --rob ./mobius_teeth.obj
# ./blender_mobius.sh -- --overlay_from_all --rob ./mobius_upper_tooth_surface.obj "$@"

./blender_mobius.sh "$@" -- \
    --sweeping_rob ./mobius_cylinder_tooth.obj \
    --enable_sweeping_rob \
	--selective_frames 187 344 556 712 849 1054 1296 \
	--grouping_selective 2 \
    --transparent_objects prev_rob \
    --resolution_x 960 --resolution_y 540 \
