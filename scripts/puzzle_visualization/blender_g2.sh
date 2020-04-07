#!/bin/bash

./facade.py tools blender --current_trial 21 --dir condor.u3/duet-g2 --puzzle_name duet-g2 \
	--camera_origin 60 -75 50 \
	--camera_lookat 15 20 0 \
	--camera_up 0 0 1 \
	--floor_origin 0 0 -20 \
	--light_auto \
	--saveas blender/blender_g2.blend \
	--animation_end 644 \
	--animation_floor_origin 0 0 -20 \
	--flat_env \
	--use_unoptimized \
	--env_camera_rotation_axis 0 1 0 \
	--rob_camera_rotation_axis 0 0 1 \
	"$@" \
