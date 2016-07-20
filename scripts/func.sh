#!/bin/bash

meshgen () {
	$BIN/meshgen -margin 0.0625 < "$1" 2>/dev/null
}

mkgen() {
	if [ "$REMESH" == "1" ]
	then
		$BIN/mkgen -b $BIN obs*.obj > mk
	else
		$BIN/mkgen -b $BIN -n obs*.obj > mk
	fi
}

run() {
	make -f mk -j `nproc`
}
