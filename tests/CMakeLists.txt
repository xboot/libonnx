if (BUILD_TESTS)
    enable_testing()

    add_executable(libonnx_test main.c)
    target_link_libraries(libonnx_test libonnx_lib)

    add_test(libonnx_test_model libonnx_test "${CMAKE_SOURCE_DIR}/tests/model")
    add_test(libonnx_test_node libonnx_test "${CMAKE_SOURCE_DIR}/tests/node")
    add_test(libonnx_test_pytorch-converted libonnx_test "${CMAKE_SOURCE_DIR}/tests/pytorch-converted")
    add_test(libonnx_test_pytorch-operator libonnx_test "${CMAKE_SOURCE_DIR}/tests/pytorch-operator")
    add_test(libonnx_test_simple libonnx_test "${CMAKE_SOURCE_DIR}/tests/simple")

endif(BUILD_TESTS)
