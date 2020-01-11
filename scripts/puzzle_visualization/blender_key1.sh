#!/bin/bash

./facade.py tools blender --current_trial 56 condor.u3/key_1_rec2/ --puzzle_name key_1_rec2 \
	--camera_origin -120 -150 -100 \
	--camera_lookat -20 0 0 \
	--camera_up -1.0 0.0 0.0 \
	--floor_origin 25 0 0 \
	--floor_euler -75 85 -35 \
	--light_auto \
	--saveas blender/blender_key1.blend \
	"$@" \
	~/HGSTArchive/blenderout/
