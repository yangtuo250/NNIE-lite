cmake_minimum_required(VERSION 3.6)
project(NNIE-lite)

SET(CMAKE_CXX_STANDARD 11)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE DEBUG)
ENDIF()
MESSAGE(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

# ------------------------------- FLAGS -----------------------------
SET(CMAKE_C_COMPILER aarch64-himix100-linux-gcc)
SET(CMAKE_CXX_COMPILER aarch64-himix100-linux-g++)
SET(CMAKE_C_STRIP aarch64-himix100-linux-strip)

# SET(CMAKE_C_FLAGS_PUBLIC "-mcpu=cortex-a8 -mfloat-abi=softfp -mfpu=neon-vfpv4 -ffunction-sections -mno-unaligned-access -fno-aggressive-loop-optimizations -mapcs-frame -rdynamic -fopenmp")
SET(CMAKE_C_FLAGS_PUBLIC "-ffunction-sections -fno-aggressive-loop-optimizations -rdynamic -fopenmp")
SET(CMAKE_C_FLAGS_DEBUG "-Wall -ggdb3 -DNM_DEBUG ${CMAKE_C_FLAGS_PUBLIC}")
SET(CMAKE_C_FLAGS_RELEASE  "-Wall -O3  ${CMAKE_C_FLAGS_PUBLIC}")
SET(CMAKE_CXX_FLAGS_PUBLIC ${CMAKE_C_FLAGS_PUBLIC})
SET(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
SET(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})


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
# GET_FILENAME_COMPONENT(cls_cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/GeneralCls.cpp ABSOLUTE)
# GET_FILENAME_COMPONENT(seg_cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/GeneralSeg.cpp ABSOLUTE)
GET_FILENAME_COMPONENT(yolo_cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/yolov5.cpp ABSOLUTE)
LIST(REMOVE_ITEM SRC ${cls_cpp} ${seg_cpp} ${yolo_cpp}) 
MESSAGE(STATUS ${SRC})
# SET(NET_SRC ./src/Net.cpp)

# # For YOLO
IF(CMAKE_BUILD_TYPE STREQUAL DEBUG)
  ADD_DEFINITIONS(-D __DEBUG__)
  ADD_EXECUTABLE(yolov5 ${SRC} ${yolo_cpp})
  TARGET_LINK_LIBRARIES(yolov5  ${HI_LIBS} ${OPENCV_LIBS} -lstdc++ -lpthread -lm -ldl )
ELSE()
  ADD_LIBRARY(yolov5 SHARED ${SRC} ${yolo_cpp})
  TARGET_LINK_LIBRARIES(yolov5 ${HI_LIBS} ${OPENCV_LIBS} -lstdc++ -lpthread -lm -ldl )
ENDIF()


# # For MNIST
# ADD_EXECUTABLE(Cls_Exe ${SRC} ./inc/GeneralCls.h ./src/GeneralCls.cpp)
# TARGET_LINK_LIBRARIES(Cls_Exe  ${HI_LIBS} ${OPENCV_LIBS} -lstdc++ -lpthread -lm -ldl )


# # For ENet
# IF(CMAKE_BUILD_TYPE STREQUAL DEBUG)
#   ADD_DEFINITIONS(-D __DEBUG__)
#   ADD_EXECUTABLE(Seg_Exe ${SRC} ./inc/GeneralSeg.h ./src/GeneralSeg.cpp)
#   TARGET_LINK_LIBRARIES(Seg_Exe  ${HI_LIBS} ${OPENCV_LIBS} -lstdc++ -lpthread -lm -ldl )
# ELSE()
#   ADD_LIBRARY(seg SHARED ${SRC} ./inc/GeneralSeg.h ./src/GeneralSeg.cpp)
#   TARGET_LINK_LIBRARIES(seg ${HI_LIBS} ${OPENCV_LIBS} -lstdc++ -lpthread -lm -ldl )
# ENDIF()
