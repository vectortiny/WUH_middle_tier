#!/bin/bash
DATE=$(date +"%Y-%m-%d_%H%M%S")

fswebcam -c ${1}/conf/fswebcam.conf ${1}/webcam/hand_${2}_${3}.jpg
