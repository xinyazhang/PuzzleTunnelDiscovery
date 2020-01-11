#!/bin/bash

./facade.py tools blender --current_trial 22 condor.u3/alpha-1.0 --puzzle_name alpha-1.0 \
	--camera_origin 25 75 155 \
	--camera_lookat 25 10 0 \
	--camera_up 0 0 1 \
	--light_auto \
	--animation_end 1254 \
	--saveas blender/blender_alpha.blend \
	"$@" \
