#!/bin/bash

./facade.py tools blender --current_trial 35 --dir condor.u3/duet-g9 --puzzle_name duet-g9 \
	--camera_origin 200 150 150 \
	--camera_lookat 30 30 0 \
	--camera_up 0 0 1 \
	--floor_origin 0 0 -15 \
	--light_auto \
	--animation_end 1369 \
	--saveas blender/blender_g9.blend \
	--animation_floor_origin 0 0 -20 \
	--env_camera_rotation_axis 0 1 0 \
	--rob_camera_rotation_axis 0 0 1 \
	--flat_env \
	"$@"
