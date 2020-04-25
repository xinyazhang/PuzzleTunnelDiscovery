#!/bin/bash

./facade.py tools blender --current_trial 20 --dir condor.u3/doublealpha-1.0 --puzzle_name doublealpha-1.0 \
	--camera_origin -10 150 250 \
	--camera_lookat -10 -5 0 \
	--camera_up 0 -1 0 \
	--light_auto \
	--floor_origin 0 0 -40 \
	--animation_end 1247 \
	--animation_floor_origin 0 0 -40 \
	--saveas blender/blender_doublealpha.blend \
	"$@" \
    -- \
    --remove_vn env rob \
    --remove_ao env rob \
