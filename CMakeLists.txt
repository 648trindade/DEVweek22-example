cmake_minimum_required(VERSION 3.23)
project(untitled CXX CUDA)

if ("$ENV{TYPE}" STREQUAL "0")

    # Equivalent in raw CLI
    # g++ -std=c++17 -fopenmp -c main.cpp -o main.o
    # g++ -std=c++17 -fopenmp -c kernel.cpp -o kernel.o
    # g++ main.o kernel.o -o untitled -fopenmp

    set(CMAKE_CXX_STANDARD 17)
    find_package(OpenMP REQUIRED)

    add_executable(untitled main.cpp kernel.cpp kernel.h)
    target_link_libraries(untitled PUBLIC OpenMP::OpenMP_CXX)

    target_include_directories(untitled SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/include)

elseif ("$ENV{TYPE}" STREQUAL "1")

    # Equivalent in raw CLI
    # g++ -std=c++17 -fopenmp -c main.cpp -o main.o
    # nvcc -std=c++17 -gencode=arch=compute_86,code=sm_86 -Xcompiler="-fopenmp" -c kernel.cu -o kernel.o
    # g++ main.o kernel.o -o untitled -fopenmp -lcudart

    set(CMAKE_CXX_STANDARD 17)
    find_package(OpenMP REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 86)

    add_executable(untitled main.cpp kernel.cu kernel.h)

    # add openmp to GCC compilation and link
    target_link_libraries(untitled PUBLIC OpenMP::OpenMP_CXX)
    # add openmp to NVCC compilation of host code
    target_compile_options(untitled PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${OpenMP_CXX_FLAGS}>)

else()

    # add standard to GCC compilation
    set(CMAKE_CXX_STANDARD 17)
    find_package(OpenMP REQUIRED)
    find_package(CUDA 11.7)

    # add openmp to GCC and NVCC compilation of host code.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # Add the standard flag to NVCC compilation of device code. Specify NVIDIA architectures
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -gencode=arch=compute_86,code=sm_86")

    cuda_add_executable(untitled main.cpp kernel.cu kernel.h)

endif ()