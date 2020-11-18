#include <onnx.h>

static int Squeeze_init(struct onnx_node_t * n)
{
	if((n->ninput >= 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Squeeze_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Squeeze_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * a;
	int64_t * pa;
	int dims[x->ndim];
	int ndim = 0;
	int axis, flag;
	int i, j;

	if(n->ninput > 1)
	{
		a = n->inputs[1];
		pa = (int64_t *)a->datas;
		for(i = 0, ndim = 0; i < x->ndim; i++)
		{
			if(x->dims[i] > 1)
				dims[ndim++] = x->dims[i];
			else
			{
				for(j = 0, flag = 0; j < a->ndata; j++)
				{
					axis = pa[j];
					if(axis < 0)
						axis += x->ndim;
					if(i == axis)
					{
						flag = 1;
						break;
					}
				}
				if(!flag)
					dims[ndim++] = x->dims[i];
			}
		}
	}
	else
	{
		for(i = 0, ndim = 0; i < x->ndim; i++)
		{
			if(x->dims[i] > 1)
				dims[ndim++] = x->dims[i];
		}
	}
	return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static void Squeeze_operator(struct onnx_node_t * n)
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
		memcpy(y->datas, x->datas, x->ndata * onnx_tensor_type_sizeof(x->type));
	}
}

void resolver_default_op_Squeeze(struct onnx_node_t * n)
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
			n->init = Squeeze_init;
			n->exit = Squeeze_exit;
			n->reshape = Squeeze_reshape;
			n->operator = Squeeze_operator;
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
