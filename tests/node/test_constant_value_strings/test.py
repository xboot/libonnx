#!/usr/bin/env python3

import onnx
import onnx.helper
import numpy as np

values = ['a','b','c','d','cat','tiger',]

output_tensor = onnx.helper.make_tensor("values", onnx.TensorProto.STRING, [len(values)], [bytes(i, 'utf-8') for i in values])

outputs = [onnx.helper.make_tensor_value_info("values", onnx.TensorProto.STRING, [len(values)])]
node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['values'],
    value_strings=values
)

graph = onnx.helper.make_graph(
    [node],
    'test-constant',
    [],
    outputs)
model = onnx.helper.make_model(graph, producer_name='onnx-example')
onnx.save_model(model, 'model.onnx')
onnx.save_tensor(output_tensor, 'test_data_set_0/output_0.pb')
