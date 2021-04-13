project( libonnx )
cmake_minimum_required( VERSION 3.19 )

option(BUILD_EXAMPLES "Build the examples" OFF)
option(BUILD_TESTS    "Build the tests" OFF)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)
