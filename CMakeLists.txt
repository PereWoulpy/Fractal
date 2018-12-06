cmake_minimum_required(VERSION 3.10.2)
project(Fractal)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CUDA_NVCC_FLAGS "-std=c++11 -lineinfo")

set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -lX11")

cuda_compile(cudaCompile cudaFractal.cu)

add_library(fractalLib gfx.cpp gfx.h utils.h cudaFractal.h)

cuda_add_executable(fractal main.cpp ${cudaCompile})
target_link_libraries(fractal fractalLib)