#!/bin/bash

./facade.py tools blender --current_trial 21 condor.u3/duet-g1 --puzzle_name duet-g1 \
	--camera_origin 5 -60 -70 \
	--camera_lookat 5 10 0 \
	--camera_up 0 0 -1 \
	--floor_origin 0 0 20 \
	--floor_euler 0 0 180 \
	--light_auto \
	--light_panel_origin 60 -20 60 \
	--light_panel_lookat 5 10 0 \
	--light_panel_up 0 0 1 \
	--saveas blender/blender_g1.blend \
	"$@" \
	--flat_env \
	~/HGSTArchive/blenderout/
