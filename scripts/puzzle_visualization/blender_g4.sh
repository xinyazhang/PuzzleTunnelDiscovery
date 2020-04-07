#!/bin/bash

./facade.py tools blender --current_trial 21 --dir condor.u3/duet-g4 --puzzle_name duet-g4 \
	--camera_origin 80 -150 100 \
	--camera_lookat 20 20 0 \
	--camera_up 0 0 1 \
	--floor_origin 0 0 -15 \
	--light_auto \
	--image_frame 147 \
	--saveas blender/blender_g4.blend \
	--animation_end 890 \
	--animation_floor_origin 0 0 -20 \
	"$@" \
	--flat_env \
