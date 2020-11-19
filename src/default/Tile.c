#include <onnx.h>

static int Tile_init(struct onnx_node_t * n)
{
	if((n->ninput == 2) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Tile_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Tile_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * r = n->inputs[1];
	int64_t * pr = (int64_t *)r->datas;
	int ndim = x->ndim;
	int dims[ndim];
	int i;

	for(i = 0; i < ndim; i++)
		dims[i] = x->dims[i] * pr[i];
	return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static void Tile_bool(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * px = (uint8_t *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_int8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int8_t * py = (int8_t *)y->datas;
	int8_t * px = (int8_t *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_int16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int16_t * py = (int16_t *)y->datas;
	int16_t * px = (int16_t *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_int32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int32_t * py = (int32_t *)y->datas;
	int32_t * px = (int32_t *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_int64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int64_t * py = (int64_t *)y->datas;
	int64_t * px = (int64_t *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_uint8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * px = (uint8_t *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_uint16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px = (uint16_t *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_uint32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t * px = (uint32_t *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_uint64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * px = (uint64_t *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px = (uint16_t *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px = (uint16_t *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	float * py = (float *)y->datas;
	float * px = (float *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	double * py = (double *)y->datas;
	double * px = (double *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_complex64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	float * py = (float *)y->datas;
	float * px = (float *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Tile_complex128(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	double * py = (double *)y->datas;
	double * px = (double *)x->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Tile_string(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	char ** px = (char **)x->datas;
	char ** py = (char **)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		px = onnx_tensor_broadcast_map_address(x, y, i);
		if(py[i])
			free(py[i]);
		py[i] = strdup(px[i]);
	}
}

void resolver_default_op_Tile(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_bool;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_float64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_complex64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX128:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_complex128;
			break;
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_string;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_bool;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_float64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_complex64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX128:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_complex128;
			break;
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->operator = Tile_string;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
	}
}
