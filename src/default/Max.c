#include <onnx.h>

static int Max_init(struct onnx_node_t * n)
{
	if((n->ninput >= 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Max_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Max_reshape(struct onnx_node_t * n)
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

static void Max_int8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	int8_t * py = (int8_t *)y->datas;
	int8_t * px;
	int8_t maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = INT8_MIN; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_int16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	int16_t * py = (int16_t *)y->datas;
	int16_t * px;
	int16_t maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = INT16_MIN; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_int32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	int32_t * py = (int32_t *)y->datas;
	int32_t * px;
	int32_t maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = INT32_MIN; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_int64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	int64_t * py = (int64_t *)y->datas;
	int64_t * px;
	int64_t maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = INT64_MIN; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_uint8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * px;
	uint8_t maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = 0; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_uint16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	uint16_t maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = 0; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_uint32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t * px;
	uint32_t maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = 0; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_uint64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * px;
	uint64_t maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = 0; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float v;
	float maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = FLT_MIN; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			v = bfloat16_to_float32(*px);
			if(v > maxv)
				maxv = v;
		}
		py[i] = float32_to_bfloat16(maxv);
	}
}

static void Max_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float v;
	float maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = FLT_MIN; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			v = float16_to_float32(*px);
			if(v > maxv)
				maxv = v;
		}
		py[i] = float32_to_float16(maxv);
	}
}

static void Max_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	float * py = (float *)y->datas;
	float * px;
	float maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = FLT_MIN; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	double * py = (double *)y->datas;
	double * px;
	double maxv;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, maxv = DBL_MIN; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

void resolver_default_op_Max(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float64;
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
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float64;
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
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float64;
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
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float64;
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
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->operator = Max_float64;
			break;
		default:
			break;
		}
	}
}
