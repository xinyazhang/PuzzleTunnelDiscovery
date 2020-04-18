#!/bin/bash

./facade.py tools blender --current_trial 30 --dir condor.u3/enigma/ --puzzle_name enigma \
	--camera_origin 30 3.5 30 \
	--camera_lookat 2 3.5 1 \
	--camera_up 0 0 1 \
	--light_auto \
	--image_frame 0 \
	--saveas blender/blender_enigma.blend \
	--floor_origin 0 0 -7 \
	--animation_floor_origin 0 0 -7 \
	"$@" \
