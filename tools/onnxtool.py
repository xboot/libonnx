from collections import OrderedDict

from typing import List, Dict, Union, Optional, Tuple
import copy

import onnx
import onnx.helper
import onnx.optimizer
import onnx.shape_inference
import onnx.numpy_helper
import onnxruntime as rt

import numpy as np
import argparse
import sys
import os

TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]

def add_features_to_output(m: onnx.ModelProto, nodes: List[onnx.NodeProto]) -> None:
    """
    Add features to output in pb, so that ONNX Runtime will output them.
    :param m: the model that will be run in ONNX Runtime
    :param nodes: nodes whose outputs will be added into the graph outputs
    """
    for node in nodes:
        for output in node.output:
            m.graph.output.extend([onnx.ValueInfoProto(name=output)])

def add_all_features_to_output(m: onnx.ModelProto) -> None:
    """
    Add features to output in pb, so that ONNX Runtime will output them.
    :param m: the model that will be run in ONNX Runtime
    """
    for node in m.graph.node:
        for output in node.output:
            m.graph.output.extend([onnx.ValueInfoProto(name=output)])

def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> List[int]:
    return [dim.dim_value for dim in v.type.tensor_type.shape.dim]

def get_value_info_all(m: onnx.ModelProto, name: str) -> Optional[onnx.ValueInfoProto]:
    for v in m.graph.value_info:
        if v.name == name:
            return v
    for v in m.graph.input:
        if v.name == name:
            return v
    for v in m.graph.output:
        if v.name == name:
            return v
    return None

def get_shape(m: onnx.ModelProto, name: str) -> TensorShape:
    """
    Note: This method relies on onnx shape inference, which is not reliable. So only use it on input or output tensors
    """
    v = get_value_info_all(m, name)
    if v is not None:
        return get_shape_from_value_info_proto(v)
    raise RuntimeError('Cannot get shape of "{}"'.format(name))

def get_elem_type(m: onnx.ModelProto, name: str) -> Optional[int]:
    v = get_value_info_all(m, name)
    if v is not None:
        return v.type.tensor_type.elem_type
    return None

def get_np_type_from_elem_type(elem_type: int) -> int:
    sizes = (None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, str, np.bool,
             np.float16, np.double, np.uint32, np.uint64, np.complex64, np.complex128, np.float16)
    assert len(sizes) == 17
    size = sizes[elem_type]
    assert size is not None
    return size

def get_input_names(model: onnx.ModelProto) -> List[str]:
    input_names = list(set([ipt.name for ipt in model.graph.input]) -
                       set([x.name for x in model.graph.initializer]))
    return input_names

def add_initializers_into_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    for x in model.graph.initializer:
        input_names = [x.name for x in model.graph.input]
        if x.name not in input_names:
            shape = onnx.TensorShapeProto()
            for dim in x.dims:
                shape.dim.extend([onnx.TensorShapeProto.Dimension(dim_value=dim)])
            model.graph.input.extend(
                [onnx.ValueInfoProto(name=x.name,
                                     type=onnx.TypeProto(tensor_type=onnx.TypeProto.Tensor(elem_type=x.data_type,
                                                                                           shape=shape)))])
    return model

def generate_rand_input(model, input_shapes: Optional[TensorShapes] = None):
    if input_shapes is None:
        input_shapes = {}
    input_names = get_input_names(model)
    full_input_shapes = {ipt: get_shape(model, ipt) for ipt in input_names}
    assert None not in input_shapes
    full_input_shapes.update(input_shapes)  # type: ignore
    for key in full_input_shapes:
        if np.prod(full_input_shapes[key]) <= 0:
            raise RuntimeError(
                'The shape of input "{}" has dynamic size, '
                'please determine the input size manually by --input-shape xxx'.format(key))

    inputs = {ipt: np.array(np.random.rand(*full_input_shapes[ipt]),
                            dtype=get_np_type_from_elem_type(get_elem_type(model, ipt))) for ipt in
              input_names}
    return inputs

def forward(model, inputs=None, input_shapes: Optional[TensorShapes] = None) -> Dict[str, np.ndarray]:
    if input_shapes is None:
        input_shapes = {}
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.log_severity_level = 3
    sess = rt.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=['CPUExecutionProvider'])
    if inputs is None:
        inputs = generate_rand_input(model, input_shapes=input_shapes)
    outputs = [x.name for x in sess.get_outputs()]
    run_options = rt.RunOptions()
    run_options.log_severity_level = 3
    res = OrderedDict(zip(outputs, sess.run(outputs, inputs, run_options=run_options)))
    return res

def check(model_opt: onnx.ModelProto, model_ori: onnx.ModelProto, n_times: int = 5,
          input_shapes: Optional[TensorShapes] = None) -> bool:
    """
    Warning: Some models (e.g., MobileNet) may fail this check by a small magnitude.
    Just ignore if it happens.
    :param input_shapes: Shapes of generated random inputs
    :param model_opt: The simplified ONNX model
    :param model_ori: The original ONNX model
    :param n_times: Generate n random inputs
    """
    if input_shapes is None:
        input_shapes = {}
    onnx.checker.check_model(model_opt)
    for i in range(n_times):
        print("Checking {}/{}...".format(i, n_times))
        rand_input = generate_rand_input(model_opt, input_shapes=input_shapes)
        res_opt = forward(model_opt, inputs=rand_input)
        res_ori = forward(model_ori, inputs=rand_input)

        for name in res_opt.keys():
            if not np.allclose(res_opt[name], res_ori[name], rtol=1e-4, atol=1e-5):
                print("Tensor {} changes after simplifying. The max diff is {}.".format(
                    name, np.max(np.abs(res_opt[name] - res_ori[name]))))
                print("Note that the checking is not always correct.")
                print("After simplifying:")
                print(res_opt[name])
                print("Before simplifying:")
                print(res_ori[name])
                print("----------------")
                return False
    return True

def load_tensor_from_pbfile(pbfile):
    tensor = onnx.TensorProto()
    with open(pbfile, 'rb') as f:
        tensor.ParseFromString(f.read())
    return onnx.numpy_helper.to_array(tensor)

def save_tensor_to_pbfile(array, pbfile):
    tensor = onnx.numpy_helper.from_array(array)
    if not os.path.exists('.outputs'):
        os.makedirs('.outputs')
    with open(os.path.join('.outputs', pbfile.replace('/', '_')), 'wb') as f:
        f.write(tensor.SerializeToString())
        print('save output tensor to .outputs/' + pbfile.replace('/', '_'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='The onnx model')
    parser.add_argument('--input', help='The input tensors should be "name:file" for example "x:input.pb"', type=str, nargs='+')
    parser.add_argument('--output', help='The output tensors will be print by name', type=str, nargs='+')
    parser.add_argument('--save', help='Enable save all output tensor to protobuf', action='store_true')
    args = parser.parse_args()
    input_tensors = {}
    if args.input is not None:
        for x in args.input:
            pieces = x.split(':')
            name, pbfile = ':'.join(pieces[:-1]), load_tensor_from_pbfile(pieces[-1])
            input_tensors[name] = pbfile
    model = onnx.load(args.model)
    add_all_features_to_output(model)
    results = forward(model, inputs = input_tensors)
    if args.output is not None:
        for y in args.output:
            t = results[y]
            print("================================================================")
            print("Name:", y)
            print("Type:", t.dtype)
            print("Size:", t.size)
            print("Shape:", t.shape)
            print("Dims:", t.ndim)
            print(t)
    else:
        for k, v in results.items():
            print(k)
    if args.save:
        for k, v in results.items():
            save_tensor_to_pbfile(v, k + ".pb")

if __name__ == '__main__':
    main()
