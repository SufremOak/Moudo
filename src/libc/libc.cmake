cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Moudo)

# Add the library
add_library(${LIB} STATIC src/libc/libc.c)

# Specify include directories
target_include_directories(${LIB} PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Add any additional source files here
# target_sources(${LIB} PRIVATE src/other_source.c)
