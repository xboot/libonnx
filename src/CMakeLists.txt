file(GLOB_RECURSE LIBONNX_LIB_SOURCES ./**.c)
file(GLOB_RECURSE LIBONNX_LIB_HEADERS ./**.h)

add_library(libonnx_lib ${LIBONNX_LIB_SOURCES})
target_include_directories(libonnx_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if (APPLE)
    target_compile_options(libonnx_lib PUBLIC -I${CMAKE_OSX_SYSROOT}/usr/include/malloc/)
endif(APPLE)
