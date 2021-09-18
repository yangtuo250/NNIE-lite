# build opencv 3.4.1 on arm64 hisi development board
cmake \
-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
-DCMAKE_C_STRIP=aarch64-linux-gnu-strip \
-DCMAKE_C_FLAGS_PUBLIC="-mcpu=cortex-a8 -mfloat-abi=softfp -mfpu=neon-vfpv4 -ffunction-sections -mno-unaligned-access -fno-aggressive-loop-optimizations -mapcs-frame -rdynamic" \
-DCMAKE_C_FLAGS_DEBUG="-Wall -ggdb3 -DNM_DEBUG ${CMAKE_C_FLAGS_PUBLIC}" \
-DCMAKE_C_FLAGS_RELEASE="-Wall -O3  ${CMAKE_C_FLAGS_PUBLIC}" \
-DCMAKE_EXE_LINKER_FLAGS="-lpthread -lrt -ldl" \
-DENABLE_CXX11=ON \
-DBUILD_TIFF=OFF \
-DBUILD_JASPER=OFF \
-DBUILD_OPENEXR=OFF \
-DBUILD_WEBP=OFF \
-DBUILD_TBB=OFF \
-DBUILD_IPP_IW=OFF \
-DBUILD_ITT=OFF \
-DWITH_AVFOUNDATION=OFF \
-DWITH_CAP_IOS=OFF \
-DWITH_CAROTENE=OFF \
-DWITH_CPUFEATURES=OFF \
-DWITH_EIGEN=OFF \
-DWITH_FFMPEG=OFF \
-DWITH_GSTREAMER=OFF \
-DWITH_GTK=OFF \
-DWITH_IPP=OFF \
-DWITH_HALIDE=OFF \
-DWITH_INF_ENGINE=OFF \
-DWITH_NGRAPH=OFF \
-DWITH_OPENEXR=OFF \
-DWITH_OPENVX=OFF \
-DWITH_GDCM=OFF \
-DWITH_TBB=OFF \
-DWITH_HPX=OFF \
-DWITH_PTHREADS_PF=OFF \
-DWITH_V4L=OFF \
-DWITH_CLP=OFF \
-DWITH_OPENCL=OFF \
-DWITH_OPENCL_SVM=OFF \
-DWITH_ITT=OFF \
-DWITH_PROTOBUF=OFF \
-DWITH_IMGCODEC_HDR=OFF \
-DWITH_IMGCODEC_SUNRASTER=OFF \
-DWITH_IMGCODEC_PXM=OFF \
-DWITH_QUIRC=OFF \
-DWITH_TENGINE=OFF \
-DBUILD_SHARED_LIBS=ON \
-DBUILD_opencv_apps=OFF \
-DBUILD_ANDROID_PROJECTS=OFF \
-DBUILD_ANDROID_EXAMPLES=OFF \
-DBUILD_DOCS=OFF \
-DBUILD_EXAMPLES=OFF \
-DBUILD_PACKAGE=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_TESTS=OFF \
-DBUILD_WITH_STATIC_CRT=OFF \
-DBUILD_FAT_JAVA_LIB=OFF \
-DBUILD_ANDROID_SERVICE=OFF \
-DBUILD_JAVA=OFF \
-DENABLE_PRECOMPILED_HEADERS=OFF \
-DENABLE_FAST_MATH=OFF \
-DCV_TRACE=OFF \
-DBUILD_opencv_java=OFF \
-DBUILD_opencv_js=OFF \
-DBUILD_opencv_ts=OFF \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=OFF \
-DBUILD_opencv_dnn=ON \
-DBUILD_opencv_imgcodecs=ON \
-DBUILD_opencv_videoio=ON \
-DBUILD_opencv_calib3d=ON \
-DBUILD_opencv_flann=ON \
-DBUILD_opencv_objdetect=ON \
-DBUILD_opencv_stitching=ON \
-DBUILD_opencv_ml=ON \
-DBUILD_opencv_shape=ON \
-DBUILD_opencv_superres=ON \
-DBUILD_opencv_videostab=ON \
..
