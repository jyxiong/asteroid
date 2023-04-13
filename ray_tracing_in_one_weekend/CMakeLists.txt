cmake_minimum_required(VERSION 3.25)

project(ray_tracing_in_one_weekend LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDA REQUIRED)

add_subdirectory(3rdparty)
add_subdirectory(one_weekend)
