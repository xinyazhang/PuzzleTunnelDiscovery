#!/bin/bash

./facade.py tools blender --current_trial 35 condor.u3/duet-g9 --puzzle_name duet-g9 \
	--camera_origin 250 30 125 \
	--camera_lookat 30 30 0 \
	--camera_up 0 0 1 \
	--floor_origin 0 0 -15 \
	--light_auto \
	--saveas blender/blender_g9.blend \
	"$@" \
	--flat_env ~/HGSTArchive/blenderout/
