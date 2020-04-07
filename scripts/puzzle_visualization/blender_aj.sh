#!/bin/bash

./facade.py tools blender --current_trial 21 --dir condor.u3/aj --puzzle_name aj \
	--camera_origin -200 40 220 \
	--camera_lookat 10 40 0 \
	--camera_up 0 0 1 \
	--light_auto \
	--image_frame 485 \
	--animation_end 1267 \
	--animation_floor_origin 0 0 -60 \
	--saveas blender/blender_aj.blend \
	"$@" \
