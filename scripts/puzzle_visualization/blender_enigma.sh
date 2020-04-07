#!/bin/bash

./facade.py tools blender --current_trial 30 --dir condor.u3/enigma/ --puzzle_name enigma \
	--camera_origin 20 3.5 20 \
	--camera_lookat 2 3.5 1 \
	--camera_up 0 0 1 \
	--light_auto \
	--image_frame 0 \
	--saveas blender/blender_enigma.blend \
	--floor_origin 0 0 -5 \
	--animation_floor_origin 0 0 -7 \
	"$@" \
