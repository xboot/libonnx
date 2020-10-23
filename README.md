

***
# Libonnx
A lightweight, portable pure `C99` `onnx` `inference engine` for embedded devices with hardware acceleration support.

## Getting Started
The library's .c and .h files can be dropped into a project and compiled along with it. Before use, should be allocated `struct onnx_context_t *` and you can pass an array of `struct resolver_t *` for hardware acceleration.

The filename is path to the format of `onnx` model.

```c
struct onnx_context_t * ctx = onnx_context_alloc_from_file(filename, NULL, 0);
```

Then, you can get input and output tensor using `onnx_search_tensor` function.

```c
struct onnx_tensor_t * input = onnx_search_tensor(ctx, "input-tensor-name");
struct onnx_tensor_t * output = onnx_search_tensor(ctx, "output-tensor-name");
```

When the input tensor has been setting, you can run inference engine using `onnx_run` function and the result will putting into the output tensor.

```c
onnx_run(ctx);
```

Finally, you must free `struct onnx_context_t *` using `onnx_context_free` function.

```c
onnx_context_free(ctx);
```

## Examples

Just type `make` at the root directory, you will see a static library and some binary of [examples](examples) for usage.

```shell
cd libonnx
make
```

## Screenshots

TBD.

## Links

* [The chinese discussion posts](https://whycan.com/t_5440.html)
* [The repo of onnx spec](https://github.com/onnx/onnx)
* [The pre-trained onnx models](https://github.com/onnx/models)

## License

This library is free software; you can redistribute it and or modify it under the terms of the MIT license. See [MIT License](LICENSE) for details.

