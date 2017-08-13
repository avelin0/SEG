set(pathToProject ~/PFC/SEG )
set(pathToOpenCv ~/OpenCV-android-sdk)


cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
include_directories(${pathToOpenCv}/sdk/native/jni/include)


add_library( native-lib SHARED src/main/cpp/native-lib.cpp )
add_library( lib_opencv SHARED IMPORTED )

set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${pathToProject}/app/src/main/jniLibs/${ANDROID_ABI}/libopencv_java3.so)

find_library( log-lib log )

target_link_libraries( native-lib $\{log-lib} lib_opencv)