#!/bin/bash

./facade.py tools blender --current_trial 22 --dir condor.u3/alpha-1.0 --puzzle_name alpha-1.0 \
	--camera_origin 25 175 200 \
	--camera_lookat 25 10 0 \
	--camera_up 0 0 1 \
	--light_auto \
	--animation_end 1403 \
	--saveas blender/blender_alpha.blend \
	"$@" \
    -- \
    --remove_vn env rob \
    --remove_ao env rob \
