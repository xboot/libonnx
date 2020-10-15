#include <onnx.h>

static void Relu_float(struct onnx_node_t * n)
{
	Onnx__TensorProto * x = n->inputs[0];
	Onnx__TensorProto * y = n->outputs[0];
	int i;

	for(i = 0; i < y->n_float_data; i++)
		y->float_data[i] = x->float_data[i] > 0 ? x->float_data[i] : 0;
}

void default_resolver_op_Relu(struct onnx_node_t * n)
{
	Onnx__TensorProto * t = n->inputs[0];
	int i;

	for(i = 0; i < n->noutput; i++)
	{
		if(n->outputs[i]->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED)
			onnx_tensor_ready(n->outputs[i], t->data_type, t->n_dims, t->dims);
	}
	switch(t->data_type)
	{
	case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
		n->op = Relu_float;
		break;
	default:
		break;
	}
}
