#include <onnx.h>

static void Shape_init(struct onnx_node_t * n)
{
	struct onnx_tensor_t * t = n->inputs[0];
	int i;

	for(i = 0; i < n->noutput; i++)
	{
		if(n->outputs[i]->type == ONNX_TENSOR_TYPE_UNDEFINED)
			onnx_tensor_reinit(n->outputs[i], ONNX_TENSOR_TYPE_INT64, (int64_t[]){ t->ndim }, 1);
	}
}

static void Shape_exit(struct onnx_node_t * n)
{
}

static void Shape_operator(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int64_t * py = (int64_t *)y->datas;
	int i, l;

	for(i = 0, l = min(y->ndata, x->ndim); i < l; i++)
		py[i] = x->dims[i];
}

void default_resolver_op_Shape(struct onnx_node_t * n)
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
		n->init = Shape_init;
		n->exit = Shape_exit;
		n->op = Shape_operator;
		break;
	default:
		break;
	}
}
