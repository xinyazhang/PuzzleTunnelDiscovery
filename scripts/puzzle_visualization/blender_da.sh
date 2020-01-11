#!/bin/bash

./facade.py tools blender --current_trial 20 condor.u3/doublealpha-1.0 --puzzle_name doublealpha-1.0 \
	--camera_origin -10 75 200 \
	--camera_lookat -5 -5 0 \
	--camera_up 0 -1 0 \
	--light_auto \
	--floor_origin 0 0 -10 \
	--saveas blender/blender_doublealpha.blend \
	"$@" \
