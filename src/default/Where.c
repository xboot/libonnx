#include <onnx.h>

static int Where_init(struct onnx_node_t * n)
{
	if((n->ninput == 3) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Where_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Where_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	int i;

	if(!onnx_tensor_reshape_identity(y, n->inputs[n->ninput - 1], n->inputs[n->ninput - 1]->type))
		return 0;
	for(i = n->ninput - 2; i >= 0; i--)
	{
		if(!onnx_tensor_reshape_multi_broadcast(y, y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

static void Where_bool(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_int8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	int8_t * py = (int8_t *)y->datas;
	int8_t * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_int16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	int16_t * py = (int16_t *)y->datas;
	int16_t * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_int32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	int32_t * py = (int32_t *)y->datas;
	int32_t * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_int64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	int64_t * py = (int64_t *)y->datas;
	int64_t * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_uint8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_uint16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_uint32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_uint64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	float * py = (float *)y->datas;
	float * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	double * py = (double *)y->datas;
	double * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_complex64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	float * py = (float *)y->datas;
	float * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Where_complex128(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	double * py = (double *)y->datas;
	double * px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = onnx_tensor_broadcast_map_address(x2, y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Where_string(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	char ** py = (char **)y->datas;
	char ** px;
	uint8_t * c;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		c = onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = (char **)onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = (char **)onnx_tensor_broadcast_map_address(x2, y, i);
		if(py[i])
			free(py[i]);
		py[i] = strdup(px[i]);
	}
}

void resolver_default_op_Where(struct onnx_node_t * n)
{
	if(n->ninput == 3)
	{
		switch(n->inputs[2]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_bool;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_float64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX64:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_complex64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX128:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_complex128;
			break;
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Where_init;
			n->exit = Where_exit;
			n->reshape = Where_reshape;
			n->operator = Where_string;
			break;
		default:
			break;
		}
	}
}
