#!/bin/bash
# Labeled Depth-Image renderer

SRC="$1"
DST="$2"
LBL=0

for path in `(cd $SRC; ls -d */)`
do
	echo $path '->' $LBL
	mkdir -p $DST/$path
	PYTHONPATH=../bin python2 ../src/RL/render-depth.py "$SRC/$path" "$DST/$path" $LBL
	LBL=$((LBL+1))
done
# PYTHONPATH=../bin python2 ../src/RL/render-depth.py "$@"
