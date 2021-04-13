
if (BUILD_EXAMPLES)
    find_package(PkgConfig REQUIRED)
    find_package(SDL2 REQUIRED)
    pkg_check_modules(sdl2_gfx REQUIRED sdl2_gfx)

    add_executable(benchmark benchmark/main.c)
    target_link_libraries(benchmark libonnx_lib)

    add_executable(hello hello/main.c)
    target_link_libraries(hello libonnx_lib)

    add_executable(mnist mnist/main.c)
    target_include_directories(mnist PRIVATE ${SDL2_INCLUDE_DIRS}/.. ${sdl2_gfx_INCLUDE_DIRS}/..)
    target_link_directories(mnist PRIVATE ${sdl2_gfx_LIBRARY_DIRS})
    target_link_libraries(mnist libonnx_lib ${SDL2_LIBRARIES} ${sdl2_gfx_LIBRARIES})
endif()
