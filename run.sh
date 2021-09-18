#!/bin/bash -e

export LD_LIBRARY_PATH=/usr/local/lib:/opt/MVS/lib/aarch64:/opt/MVS/lib/aarch64
ldconfig
./release/Seg_Exe
