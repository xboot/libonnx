#include <onnx.h>

static void Size_init(struct onnx_node_t * n)
{
	int i;

	for(i = 0; i < n->noutput; i++)
	{
		if(n->outputs[i]->type == ONNX_TENSOR_TYPE_UNDEFINED)
			onnx_tensor_reinit(n->outputs[i], ONNX_TENSOR_TYPE_INT64, NULL, 0);
	}
}

static void Size_exit(struct onnx_node_t * n)
{
}

static void Size_operator(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	y->scalar.v_int64 = x->ndata;
}

void default_resolver_op_Size(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_BOOL:
	case ONNX_TENSOR_TYPE_INT8:
	case ONNX_TENSOR_TYPE_INT16:
	case ONNX_TENSOR_TYPE_INT32:
	case ONNX_TENSOR_TYPE_INT64:
	case ONNX_TENSOR_TYPE_UINT8:
	case ONNX_TENSOR_TYPE_UINT16:
	case ONNX_TENSOR_TYPE_UINT32:
	case ONNX_TENSOR_TYPE_UINT64:
	case ONNX_TENSOR_TYPE_BFLOAT16:
	case ONNX_TENSOR_TYPE_FLOAT16:
	case ONNX_TENSOR_TYPE_FLOAT32:
	case ONNX_TENSOR_TYPE_FLOAT64:
	case ONNX_TENSOR_TYPE_COMPLEX64:
	case ONNX_TENSOR_TYPE_COMPLEX128:
	case ONNX_TENSOR_TYPE_STRING:
		n->init = Size_init;
		n->exit = Size_exit;
		n->op = Size_operator;
		break;
	default:
		break;
	}
}
