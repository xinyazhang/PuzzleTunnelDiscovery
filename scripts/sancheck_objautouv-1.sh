#!/bin/bash

FN=objautouv.sc1.obj
rm -f $FN

./objautouv /dev/stdin $FN <<EOF
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 -0.5 0.0
v 0.5 0.0 -0.5
f 1 3 2
f 1 2 4
EOF
