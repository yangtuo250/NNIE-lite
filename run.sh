#!/bin/bash -e

export LD_LIBRARY_PATH=/usr/local/lib:/opt/MVS/lib/aarch64
ldconfig
rm /root/data/tile/*result*
./release/yolov5
