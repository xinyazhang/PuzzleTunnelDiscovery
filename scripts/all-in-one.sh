#!/bin/bash

INITIAL_DIR=`pwd`
SRCDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIN=$SRCDIR/../bin

echo $INITIAL_DIR
echo $SRCDIR
echo $BIN

source $SRCDIR/func.sh

REMESH=1
maze_file=""
output_dir=""

while getopts "nf:d:" opt; do
	case $opt in
		n)	REMESH=0
			;;
		f)	maze_file=$OPTARG
			;;
		d)	output_dir=$OPTARG
			;;
		\?)
			echo "Invalid option: -$OPTARG" >&2
			;;
	esac
done

if [ "$maze_file" == "" ]
then
	echo "-f is necessary as maze file input"
	exit 0;
fi

if [ "$output_dir" == "" ]
then
	echo "-d is necessary to specify output directory"
	exit 0;
fi

mkdir -p $output_dir
cd $output_dir
meshgen $INITIAL_DIR/$maze_file
mkgen
run
