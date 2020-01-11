#!/bin/bash

./facade.py tools blender --current_trial 21 condor.u3/duet-g2 --puzzle_name duet-g2 \
	--camera_origin 60 -75 50 \
	--camera_lookat 15 20 0 \
	--camera_up 0 0 1 \
	--floor_origin 0 0 -20 \
	--light_auto \
	--saveas blender/blender_g2.blend \
	"$@" \
	--flat_env \
