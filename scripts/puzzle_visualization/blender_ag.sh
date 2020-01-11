#!/bin/bash

./facade.py tools blender --current_trial 21 condor.u3/ag-2 --puzzle_name ag-2 \
	--camera_origin -75 -20 200 \
	--camera_lookat 20 -20 0 \
	--camera_up 0 0 1 \
	--light_auto \
	--saveas blender/blender_ag.blend \
	"$@" \
	~/HGSTArchive/blenderout/
