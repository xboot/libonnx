

***
# Libonnx
A lightweight, portable pure `C99` `onnx` `inference engine` for embedded devices with hardware acceleration support.

## Getting Started
The library's .c and .h files can be dropped into a project and compiled along with it. Before use, should be allocated `struct onnx_context_t *` and you can pass an array of `struct resolver_t *` for hardware acceleration.

The filename is path to the format of `onnx` model.

```c
struct onnx_context_t * ctx = onnx_context_alloc_from_file(filename, NULL, 0);
```

Then, you can get input and output tensor using `onnx_tensor_search` function.

```c
struct onnx_tensor_t * input = onnx_tensor_search(ctx, "input-tensor-name");
struct onnx_tensor_t * output = onnx_tensor_search(ctx, "output-tensor-name");
```

When the input tensor has been setting, you can run inference engine using `onnx_run` function and the result will putting into the output tensor.

```c
onnx_run(ctx);
```

Finally, you must free `struct onnx_context_t *` using `onnx_context_free` function.

```c
onnx_context_free(ctx);
```

## Compilation Instructions

Just type `make` at the root directory, you will see a static library and some binary of [examples](examples) and [tests](tests) for usage.

```shell
cd libonnx
make
```

To compile the `mnist` example, you will have to install SDL2 and SDL2 GFX. On systems like Ubuntu run
```shell
    apt-get install libsdl2-dev libsdl2-gfx-dev
```
to install the required Simple DirectMedia Layer libraries to run the GUI.

#### Cross compilation example (for `arm64`)

Run `make CROSS_COMPILE=path/to/toolchains/aarch64-linux-gnu-` at the root directory to compile all libraries, tests and examples for the platform.

Change `CROSS_COMPILE` to point the toolchains that you plan to use.


#### How to run examples

After compiling all the files, you can run an example by using:

```shell
cd libonnx/examples/hello
./hello
```

## Screenshots
* [Mnist handwritten digit prediction](examples/mnist)
![Mnist handwritten digit prediction](documents/images/mnist.gif)

## Running tests

To run tests, for example on those in the `tests/model` folder use:

```shell
cd libonnx/tests/
./tests model
```

Here is the output:
```shell
[mnist_8](test_data_set_0)                                                              [OKAY]
[mnist_8](test_data_set_1)                                                              [OKAY]
[mnist_8](test_data_set_2)                                                              [OKAY]
[mobilenet_v2_7](test_data_set_0)                                                       [OKAY]
[mobilenet_v2_7](test_data_set_1)                                                       [OKAY]
[mobilenet_v2_7](test_data_set_2)                                                       [OKAY]
[shufflenet_v1_9](test_data_set_0)                                                      [OKAY]
[shufflenet_v1_9](test_data_set_1)                                                      [OKAY]
[shufflenet_v1_9](test_data_set_2)                                                      [OKAY]
[squeezenet_v11_7](test_data_set_0)                                                     [OKAY]
[squeezenet_v11_7](test_data_set_1)                                                     [OKAY]
[squeezenet_v11_7](test_data_set_2)                                                     [OKAY]
[super_resolution_10](test_data_set_0)                                                  [OKAY]
[tinyyolo_v2_8](test_data_set_0)                                                        [OKAY]
[tinyyolo_v2_8](test_data_set_1)                                                        [OKAY]
[tinyyolo_v2_8](test_data_set_2)                                                        [OKAY]
```

Note that running the test on the other folders may not succeed. Some operators have not been implemented, look bat the notes section for more info.

## Notes

- This library based on the onnx version [v1.17.0](https://github.com/onnx/onnx/tree/v1.17.0) with the newest `opset 23` support. [The supported operator table](documents/the-supported-operator-table.md) in the [documents](documents) directory.
- Checkout the `tools` folder for help with ONNX model files.
- You can use `xxd -i <filename.onnx>` (on Linux) to convert your onnx model into a `unsigned char array` and then use the function `onnx_context_alloc` to use it. This is how the models are loaded in the examples - `hello` and `mnist`.

## Links

* [The chinese discussion posts](https://whycan.com/t_5440.html)
* [The onnx operators documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md)
* [The tutorials for creating ONNX models](https://github.com/onnx/tutorials)
* [The pre-trained onnx models](https://github.com/onnx/models)

## License

This library is free software; you can redistribute it and or modify it under the terms of the MIT license. See [MIT License](LICENSE) for details.

