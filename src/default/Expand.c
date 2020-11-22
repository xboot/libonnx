#include <onnx.h>

static int Expand_init(struct onnx_node_t * n)
{
	if((n->ninput == 2) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Expand_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Expand_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * s = n->inputs[1];
	int64_t * ps = (int64_t *)s->datas;
	int ndim = max(x->ndim, (int)s->ndata);
	int dims[ndim];
	int i, j, k;

	for(i = x->ndim - 1, j = s->ndata - 1, k = ndim - 1; k >= 0; k--)
	{
		if(i < 0)
			dims[k] = ps[j--];
		else if(j < 0)
			dims[k] = x->dims[i--];
		else
		{
			if(x->dims[i] == ps[j])
				dims[k] = x->dims[i];
			else if((x->dims[i] == 1) || (ps[j] == 1))
				dims[k] = (x->dims[i] > ps[j]) ? x->dims[i] : ps[j];
			else
				return 0;
			i--;
			j--;
		}
	}
	return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static void Expand_bool(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * px = (uint8_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_int8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int8_t * py = (int8_t *)y->datas;
	int8_t * px = (int8_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_int16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int16_t * py = (int16_t *)y->datas;
	int16_t * px = (int16_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_int32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int32_t * py = (int32_t *)y->datas;
	int32_t * px = (int32_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_int64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int64_t * py = (int64_t *)y->datas;
	int64_t * px = (int64_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_uint8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * px = (uint8_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_uint16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px = (uint16_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_uint32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t * px = (uint32_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_uint64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * px = (uint64_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px = (uint16_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px = (uint16_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	float * py = (float *)y->datas;
	float * px = (float *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	double * py = (double *)y->datas;
	double * px = (double *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Expand_complex64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	float * py = (float *)y->datas;
	float * px = (float *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Expand_complex128(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	double * py = (double *)y->datas;
	double * px = (double *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Expand_string(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	char ** px = (char **)x->datas;
	char ** py = (char **)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		if(py[i])
			free(py[i]);
		py[i] = strdup(px[i]);
	}
}

void resolver_default_op_Expand(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_bool;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_float64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX64:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_complex64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX128:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_complex128;
			break;
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_string;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 8)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_bool;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_float64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX64:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_complex64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX128:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_complex128;
			break;
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Expand_init;
			n->exit = Expand_exit;
			n->reshape = Expand_reshape;
			n->operator = Expand_string;
			break;
		default:
			break;
		}
	}
}
