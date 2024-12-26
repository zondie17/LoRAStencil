cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(LoRAStencil LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
enable_language(CXX CUDA)

# Define the include DIRs
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

# Define the link libraries
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(/usr/local/cuda/lib64)
add_executable(lorastencil_1d src/1d/main.cu src/1d/gpu_1r.cu src/1d/gpu_2r.cu src/1d/1d_utils.h)
set_target_properties(
        lorastencil_1d
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80")
target_compile_options(lorastencil_1d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math --gpu-architecture=sm_80  --ptxas-options=-v  -lineinfo>)

add_executable(lorastencil_2d src/2d/main.cu src/2d/gpu.cu src/2d/2d_utils.h)
set_target_properties(
        lorastencil_2d
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80")
target_compile_options(lorastencil_2d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math --gpu-architecture=sm_80 --ptxas-options=-v  -lineinfo>)

add_executable(lorastencil_3d src/3d/main.cu src/3d/gpu_box.cu src/3d/gpu_star.cu  src/3d/3d_utils.h)
set_target_properties(
        lorastencil_3d
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80")
target_compile_options(lorastencil_3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math --gpu-architecture=sm_80 --ptxas-options=-v -lineinfo>)

set(CMAKE_CUDA_ARCHITECTURES 80)