# build opencv 3.4.4 on X86 for nnie11
cmake \
-DBUILD_TIFF=ON \
-DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_INSTALL_PREFIX=/home/liuyk/miniconda3/envs/nnie11 \
-DPYTHON_EXECUTABLE=/home/liuyk/miniconda3/envs/nnie11/bin/python \
-DBUILD_SHARED_LIBS=ON \
-DPYTHON_LIBRARY=/home/liuyk/miniconda3/envs/nnie11/lib/libpython2.7.so \
-DBUILD_PTYHON3=OFF \
-DCMAKE_CXX_FLAGS="-std=c++11 -O3" \
..
