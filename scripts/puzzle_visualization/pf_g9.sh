#!/bin/bash

# ./blender_g9.sh -- --overlay_from 67 158 242 322 405 492 510 539 627 754 772 874 968 1060 1114 1193 1322 1353
# ./blender_g9.sh -- --overlay_from 67 158 242 322 405 492 510 539 --rob ./knot.dt.tcp.obj
# ./blender_g9.sh -- --rob ./knot.dt.tcp.obj
# ./blender_mobius.sh -- --overlay_from_all --rob ./mobius_teeth.obj
# ./blender_mobius.sh -- --overlay_from_all --rob ./mobius_upper_tooth_surface.obj "$@"

./blender_g9.sh "$@" \
	--camera_origin 150 30 200 \
    -- \
    --sweeping_rob ./duet_cylinder_tooth.obj \
    --enable_sweeping_rob \
	--selective_frames 65 87 158 242 324 405 441 490 510 537 593 629 665 749 773 820 873 967 1021 1061 1084 1110 1148 1191 1280 1351 \
	--grouping_selective 3 \
    --transparent_objects prev_rob \
    --disable_intermediate_rob \
    --resolution_x 960 --resolution_y 540 \
