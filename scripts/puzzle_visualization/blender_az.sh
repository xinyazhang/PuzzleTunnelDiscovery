#!/bin/bash

./facade.py tools blender --current_trial 21 --dir condor.u3/az --puzzle_name az \
	--camera_origin 15 -150 200 \
	--camera_lookat 15 20 0 \
	--camera_up 0 0 1 \
	--light_auto \
	--animation_end 912 \
	--saveas blender/blender_az.blend \
	"$@" \
    -- \
    --remove_vn env rob \
    --remove_ao env rob \
