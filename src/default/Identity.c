#include <onnx.h>

static int Identity_init(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x;
	struct onnx_tensor_t * y;

	if((n->ninput > 0) && (n->noutput > 0))
	{
		x = n->inputs[0];
		y = n->outputs[0];
		if(!onnx_tensor_shape_equal(y, x) || (y->type != x->type))
			onnx_tensor_reinit(y, x->type, x->dims, x->ndim);
		return 1;
	}
	return 0;
}

static int Identity_exit(struct onnx_node_t * n)
{
	return 1;
}

static void Identity_operator(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	char ** px = (char **)x->datas;
	char ** py = (char **)y->datas;
	int i, l;

	if(x->type == ONNX_TENSOR_TYPE_STRING)
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			if(py[i])
				free(py[i]);
			py[i] = strdup(px[i]);
		}
	}
	else
	{
		if(x->ndim > 0)
			memcpy(y->datas, x->datas, x->ndata * onnx_tensor_type_sizeof(x->type));
		else
			memcpy(&y->scalar, &x->scalar, sizeof(union onnx_scalar_t));
	}
}

void resolver_default_op_Identity(struct onnx_node_t * n)
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
		n->init = Identity_init;
		n->exit = Identity_exit;
		n->op = Identity_operator;
		break;
	default:
		break;
	}
}
