#!/bin/bash

./facade.py tools blender --current_trial 38 condor.u3/mobius --puzzle_name mobius \
	--camera_origin -175 -125 -20 \
	--camera_lookat 0 0 0 \
	--camera_up 0.8 0.99 0.1 \
	--floor_origin 20 0 0 \
	--floor_euler 0 90 0 \
	--light_auto \
	--image_frame 398 \
	--saveas blender/blender_mobius.blend \
	--animation_end 1366 \
	--animation_floor_origin 50 0 0 \
	"$@" \
