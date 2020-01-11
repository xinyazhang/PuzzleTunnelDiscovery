#!/bin/bash

./facade.py tools blender --current_trial 21 condor.u3/aj --puzzle_name aj \
	--camera_origin 15 -90 150 \
	--camera_lookat 15 20 0 \
	--camera_up 0 0 1 \
	--light_auto \
	--image_frame 485 \
	--saveas blender/blender_aj.blend \
	"$@" \
