#include <onnx.h>

static void Relu_init(struct onnx_node_t * n)
{
	Onnx__TensorProto * t = n->inputs[0];
	int i;

	for(i = 0; i < n->noutput; i++)
	{
		if(n->outputs[i]->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED)
			onnx_tensor_ready(n->outputs[i], t->data_type, t->n_dims, t->dims);
	}
}

static void Relu_exit(struct onnx_node_t * n)
{
}

static void Relu_float(struct onnx_node_t * n)
{
	Onnx__TensorProto * x = n->inputs[0];
	Onnx__TensorProto * y = n->outputs[0];
	int i, l;

	for(i = 0, l = y->n_float_data; i < l; i++)
		y->float_data[i] = (x->float_data[i] < 0) ? 0 : x->float_data[i];
}

static void Relu_float16(struct onnx_node_t * n)
{
	Onnx__TensorProto * x = n->inputs[0];
	Onnx__TensorProto * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->int32_data;
	uint16_t * py = (uint16_t *)y->int32_data;
	float v;
	int i, l;

	for(i = 0, l = (y->n_int32_data << 1); i < l; i++)
	{
		v = float16_to_float32(px[i]);
		if(v < 0)
			v = 0;
		py[i] = float32_to_float16(v);
	}
}

static void Relu_double(struct onnx_node_t * n)
{
	Onnx__TensorProto * x = n->inputs[0];
	Onnx__TensorProto * y = n->outputs[0];
	int i, l;

	for(i = 0, l = y->n_double_data; i < l; i++)
		y->double_data[i] = (x->double_data[i] < 0) ? 0 : x->double_data[i];
}

static void Relu_bfloat16(struct onnx_node_t * n)
{
	Onnx__TensorProto * x = n->inputs[0];
	Onnx__TensorProto * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->int32_data;
	uint16_t * py = (uint16_t *)y->int32_data;
	float v;
	int i, l;

	for(i = 0, l = (y->n_int32_data << 1); i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		if(v < 0)
			v = 0;
		py[i] = float32_to_bfloat16(v);
	}
}

void default_resolver_op_Relu(struct onnx_node_t * n)
{
	switch(n->inputs[0]->data_type)
	{
	case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
		n->init = Relu_init;
		n->exit = Relu_exit;
		n->op = Relu_float;
		break;
	case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
		n->init = Relu_init;
		n->exit = Relu_exit;
		n->op = Relu_float16;
		break;
	case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
		n->init = Relu_init;
		n->exit = Relu_exit;
		n->op = Relu_double;
		break;
	case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
		n->init = Relu_init;
		n->exit = Relu_exit;
		n->op = Relu_bfloat16;
		break;
	default:
		break;
	}
}
