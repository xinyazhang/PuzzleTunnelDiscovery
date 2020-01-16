#!/bin/bash

./facade.py tools blender --current_trial 56 condor.u3/key_1_rec2/ --puzzle_name key_1_rec2 \
	--camera_origin -187 -225 -5 \
	--camera_lookat -25 -25 -5 \
	--camera_up -1.0 0.0 0.0 \
	--floor_origin 100 0 0 \
	--floor_euler -75 85 -35 \
	--light_auto \
	--saveas blender/blender_key1.blend \
	--animation_floor_origin 100 0 0 \
	"$@" \
