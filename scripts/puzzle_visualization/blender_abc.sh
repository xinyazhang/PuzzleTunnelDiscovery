#!/bin/bash

./facade.py tools blender --current_trial 50 condor.u3/abc_rec2m/ --puzzle_name abc_rec2m \
	--camera_origin -150 -100 -10 \
	--camera_lookat 0 0 0 \
	--camera_up 0.8 0.99 0.1 \
	--floor_origin 40 0 0 \
	--floor_euler 0 90 0 \
	--light_auto \
	--saveas blender/abc.blend \
	"$@" \
	~/HGSTArchive/blenderout/
