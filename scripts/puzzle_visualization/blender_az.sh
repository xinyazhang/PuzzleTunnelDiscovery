#!/bin/bash

./facade.py tools blender --current_trial 21 condor.u3/az --puzzle_name az \
	--camera_origin 15 -90 150 \
	--camera_lookat 15 30 0 \
	--camera_up 0 0 1 \
	--light_auto \
	--saveas blender/blender_az.blend \
	"$@" \
