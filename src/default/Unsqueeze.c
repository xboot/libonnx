#include "../onnx.h"

static int Unsqueeze_init(struct onnx_node_t * n)
{
	if((n->ninput == 2) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Unsqueeze_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Unsqueeze_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * a = n->inputs[1];
	int64_t * pa = (int64_t *)a->datas;
	int ndim = x->ndim + a->ndata;
	int dims[ndim];
	int axis;
	int i, j;

	memset(dims, 0, sizeof(int) * ndim);
	for(i = 0; i < a->ndata; i++)
	{
		axis = pa[i];
		if(axis < 0)
			axis += ndim;
		if(axis >= 0 && axis < ndim)
			dims[axis] = 1;
	}
	for(i = 0, j = 0; i < ndim; i++)
	{
		if(dims[i] != 1)
			dims[i] = x->dims[j++];
	}
	return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static void Unsqueeze_operator(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	char ** px = (char **)x->datas;
	char ** py = (char **)y->datas;

	if(x->type == ONNX_TENSOR_TYPE_STRING)
	{
		size_t i,l;
		for(i=0, l = y->ndata; i < l; i++)
		{
			if(py[i])
				free(py[i]);
			py[i] = strdup(px[i]);
		}
	}
	else
	{
		memcpy(y->datas, x->datas, x->ndata * onnx_tensor_type_sizeof(x->type));
	}
}

void resolver_default_op_Unsqueeze(struct onnx_node_t * n)
{
	if(n->opset >= 13)
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
			n->init = Unsqueeze_init;
			n->exit = Unsqueeze_exit;
			n->reshape = Unsqueeze_reshape;
			n->operator = Unsqueeze_operator;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 11)
	{
	}
	else if(n->opset >= 1)
	{
	}
}
