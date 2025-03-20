# CythonConfig.cmake

# Locate the Cython executable
find_program(CYTHON_EXECUTABLE NAMES cython)

if(NOT CYTHON_EXECUTABLE)
    message(FATAL_ERROR "Cython executable not found. Please install Cython.")
endif()

# Function to add a Cython module
function(add_cython_module target_name source_file)
    # Generate the C++ source file from the Cython file
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.cpp
        COMMAND ${CYTHON_EXECUTABLE} -3 --cplus -o ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
        COMMENT "Generating C++ source from ${source_file}"
    )

    # Add the generated C++ source file to the target
    add_library(${target_name} MODULE ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.cpp)
    target_include_directories(${target_name} PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${PYTHON_LIBRARIES})
endfunction()