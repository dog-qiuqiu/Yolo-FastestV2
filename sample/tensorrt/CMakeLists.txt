cmake_minimum_required(VERSION 3.20.5)
# set(CMAKE_C_COMPILER /home/wxk/gcc/gcc/bin/gcc)
# set(CMAKE_CXX_COMPILER /home/wxk/gcc/gcc/bin/g++)

project(test)

add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

# 引入非标准的头文件搜索路径
include_directories(/home/wxk/tensorrt/tensorrt/include/)
include_directories(/home/wxk/cuda/cuda/include/)
include_directories(./src/include)
# 添加非标准的共享库搜索路径
link_directories(/home/wxk/tensorrt/tensorrt/lib/ /home/wxk/cuda/cuda/lib64)
link_libraries(stdc++fs)

add_executable(test ./src/musicyolo.cpp test.cpp)
target_link_libraries(test nvinfer)
target_link_libraries(test cudart)
target_link_libraries(test nvonnxparser)
target_link_libraries(test nvparsers)