cmake_minimum_required(VERSION 3.12)
project(Moudo)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if (NOT Python3_FOUND)
    set(Python3_EXECUTABLE "./.venv/bin/python")
    set(Python3_INCLUDE_DIRS "./.venv/include/python3.8")
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
endif()
find_package(Cython REQUIRED)

# Check for Windows on ARM
if (WIN32 AND CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
    message(FATAL_ERROR "Windows on ARM is not supported")
endif()

# Set the output library name based on the host GPU
if (DEFINED ENV{CUDA_VISIBLE_DEVICES})
    set(LIBRARY_NAME "LibMoudoPyCxx99GTXRTXCuda")
else()
    set(LIBRARY_NAME "LibMoudoPyCxx99")
endif()

# Set the output library extension based on the operating system
if (WIN32)
    set(LIBRARY_EXTENSION ".dll")
else()
    set(LIBRARY_EXTENSION ".so")
endif()

# Collect all .pyx files in the src directory
file(GLOB_RECURSE CYTHON_FILES src/*.pyx)

# Collect all .cu files in the src/nvidia directory
file(GLOB_RECURSE CUDA_FILES src/nvidia/*.cu)

# Enable CUDA support
find_package(CUDA REQUIRED)

# Compile .pyx files to .c
add_custom_command(
    OUTPUT ${CYTHON_FILES}
    COMMAND ${Python3_EXECUTABLE} -m cython ${CYTHON_FILES}
    DEPENDS ${CYTHON_FILES}
    COMMENT "Compiling Cython files to C"
)

# Create the shared library
add_library(${LIBRARY_NAME} SHARED ${CYTHON_FILES} ${CUDA_FILES})

# Link Python and CUDA libraries
target_link_libraries(${LIBRARY_NAME} Python3::Python CUDA::cudart)