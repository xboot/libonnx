#include <onnx.h>

static int Min_init(struct onnx_node_t * n)
{
	if((n->ninput >= 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Min_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Min_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	int i;

	if(!onnx_tensor_reshape_identity(y, n->inputs[0], n->inputs[0]->type))
		return 0;
	for(i = 1; i < n->ninput; i++)
	{
		if(!onnx_tensor_reshape_multi_broadcast(y, y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

static void Min_int8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	int8_t * py = (int8_t *)y->datas;
	int8_t * px;
	int8_t minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = INT8_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

static void Min_int16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	int16_t * py = (int16_t *)y->datas;
	int16_t * px;
	int16_t minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = INT16_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

static void Min_int32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	int32_t * py = (int32_t *)y->datas;
	int32_t * px;
	int32_t minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = INT32_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

static void Min_int64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	int64_t * py = (int64_t *)y->datas;
	int64_t * px;
	int64_t minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = INT64_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

static void Min_uint8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * px;
	uint8_t minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = UINT8_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

static void Min_uint16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	uint16_t minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = UINT16_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

static void Min_uint32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t * px;
	uint32_t minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = UINT32_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

static void Min_uint64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * px;
	uint64_t minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = UINT64_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

static void Min_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float v;
	float minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = FLT_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			v = bfloat16_to_float32(*px);
			if(v < minv)
				minv = v;
		}
		py[i] = float32_to_bfloat16(minv);
	}
}

static void Min_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float v;
	float minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = FLT_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			v = float16_to_float32(*px);
			if(v < minv)
				minv = v;
		}
		py[i] = float32_to_float16(minv);
	}
}

static void Min_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	float * py = (float *)y->datas;
	float * px;
	float minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = FLT_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

static void Min_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	double * py = (double *)y->datas;
	double * px;
	double minv;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, minv = DBL_MAX; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

void resolver_default_op_Min(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 12)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 8)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Min_init;
			n->exit = Min_exit;
			n->reshape = Min_reshape;
			n->operator = Min_float64;
			break;
		default:
			break;
		}
	}
}
