cmake_minimum_required(VERSION 3.6)
project(NNIE-lite)

SET(CMAKE_CXX_STANDARD 11)

OPTION(DEBUG "debug mode, binary executable compiled, not shared library" OFF)

# ------------------------------- FLAGS -----------------------------
SET(CMAKE_C_COMPILER aarch64-himix100-linux-gcc)
SET(CMAKE_CXX_COMPILER aarch64-himix100-linux-g++)
SET(CMAKE_C_STRIP aarch64-himix100-linux-strip)

SET(CMAKE_C_FLAGS_PUBLIC "-mcpu=cortex-a8 -mfloat-abi=softfp -mfpu=neon-vfpv4 -ffunction-sections -mno-unaligned-access -fno-aggressive-loop-optimizations -mapcs-frame -rdynamic")
SET(CMAKE_C_FLAGS_DEBUG "-Wall -ggdb3 -DNM_DEBUG ${CMAKE_C_FLAGS_PUBLIC}")
SET(CMAKE_C_FLAGS_RELEASE  "-Wall -O3  ${CMAKE_C_FLAGS_PUBLIC}")


SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../release)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../release/lib)

# --------------------------------
SET(THIRD_PART third_party)
# Hisi
SET(HI_ROOT ${THIRD_PART}/hisi)
SET(HI_INCLUDE ${HI_ROOT}/include)
SET(HI_LIB_DIR ${HI_ROOT}/lib)
# opencv
SET(OPENCV_ROOT ./third_party/opencv-3.4.1/install_hisi)
SET(OPENCV_INCLUDE ${OPENCV_ROOT}/include)
SET(OPENCV_LIB_DIR ${OPENCV_ROOT}/lib)
FILE(GLOB OPENCV_LIBS "${OPENCV_LIB_DIR}/*")

# ------------ lib ----------------
LINK_LIBRARIES(mpi.a)
SET(HI_LIBS _hiawb.a _hiae.a _hidehaze.a upvqe.a dnvqe.a md.a
        _hildci.a _hidrc.a securec.a nnie.a ive.a tde.a VoiceEngine.a isp.a )

INCLUDE_DIRECTORIES(inc  ${OPENCV_INCLUDE} ${HI_INCLUDE})
LINK_DIRECTORIES( ${OPENCV_LIB_DIR} ${HI_LIB_DIR})

FILE(GLOB SRC "src/*.cpp" "src/*c")
SET(NET_SRC ./src/Net.cpp)

# # For YOLO
# ADD_EXECUTABLE(alg_Exe ${SRC} src/yolo.cpp)
# TARGET_LINK_LIBRARIES(alg_Exe  ${HI_LIBS} ${OPENCV_LIBS} -lstdc++ -lpthread -lm -ldl )


# For MNIST
ADD_EXECUTABLE(Cls_Exe ${SRC} ./inc/GeneralCls.h ./src/GeneralCls.cpp)
TARGET_LINK_LIBRARIES(Cls_Exe  ${HI_LIBS} ${OPENCV_LIBS} -lstdc++ -lpthread -lm -ldl )


# For ENet
IF(DEBUG)
  ADD_DEFINITIONS(-D __DEBUG__)
  ADD_EXECUTABLE(Seg_Exe ${SRC} ./inc/GeneralSeg.h ./src/GeneralSeg.cpp)
  TARGET_LINK_LIBRARIES(Seg_Exe  ${HI_LIBS} ${OPENCV_LIBS} -lstdc++ -lpthread -lm -ldl )
ELSE()
  ADD_LIBRARY(libseg SHARED ${SRC} ./inc/GeneralSeg.h ./src/GeneralSeg.cpp)
  TARGET_LINK_LIBRARIES(libseg ${HI_LIBS} ${OPENCV_LIBS} -lstdc++ -lpthread -lm -ldl )
ENDIF()
