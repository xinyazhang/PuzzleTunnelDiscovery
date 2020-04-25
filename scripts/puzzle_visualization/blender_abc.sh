#!/bin/bash

./facade.py tools blender --current_trial 50 --dir condor.u3/abc_rec2m/ --puzzle_name abc_rec2m \
	--camera_origin -157.5 -105 -10.5 \
	--camera_lookat 0 0 0 \
	--camera_up 0.8 0.99 0.1 \
	--floor_origin 40 0 0 \
	--floor_euler 0 90 0 \
	--light_auto \
	--animation_end 1424 \
	--animation_floor_origin 50 0 0 \
	--mod_weighted_normal env \
	--saveas blender/abc.blend \
	"$@" \
    -- \
    --remove_ao env rob \
