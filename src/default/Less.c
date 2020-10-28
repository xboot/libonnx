#include <onnx.h>

static int Less_init(struct onnx_node_t * n)
{
	if((n->ninput == 2) && (n->noutput == 1))
	{
		struct onnx_tensor_t * a = n->inputs[0];
		struct onnx_tensor_t * b = n->inputs[1];
		struct onnx_tensor_t * y = n->outputs[0];
		if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
			return 1;
	}
	return 0;
}

static int Less_exit(struct onnx_node_t * n)
{
	return 1;
}

static void Less_int8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	int8_t * pa;
	int8_t * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (*pa < *pb) ? 1 : 0;
		}
	}
}

static void Less_int16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	int16_t * pa;
	int16_t * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (*pa < *pb) ? 1 : 0;
		}
	}
}

static void Less_int32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	int32_t * pa;
	int32_t * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (*pa < *pb) ? 1 : 0;
		}
	}
}

static void Less_int64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	int64_t * pa;
	int64_t * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (*pa < *pb) ? 1 : 0;
		}
	}
}

static void Less_uint8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * pa;
	uint8_t * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (*pa < *pb) ? 1 : 0;
		}
	}
}

static void Less_uint16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (*pa < *pb) ? 1 : 0;
		}
	}
}

static void Less_uint32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint32_t * pa;
	uint32_t * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (*pa < *pb) ? 1 : 0;
		}
	}
}

static void Less_uint64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint64_t * pa;
	uint64_t * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (*pa < *pb) ? 1 : 0;
		}
	}
}

static void Less_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (bfloat16_to_float32(*pa) < bfloat16_to_float32(*pb)) ? 1 : 0;
		}
	}
}

static void Less_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (float16_to_float32(*pa) < float16_to_float32(*pb)) ? 1 : 0;
		}
	}
}

static void Less_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	float * pa;
	float * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (*pa < *pb) ? 1 : 0;
		}
	}
}

static void Less_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	double * pa;
	double * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, ONNX_TENSOR_TYPE_BOOL))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (*pa < *pb) ? 1 : 0;
		}
	}
}

void resolver_default_op_Less(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_INT8:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_int8;
		break;
	case ONNX_TENSOR_TYPE_INT16:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_int16;
		break;
	case ONNX_TENSOR_TYPE_INT32:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_int32;
		break;
	case ONNX_TENSOR_TYPE_INT64:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_int64;
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_uint8;
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_uint16;
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_uint32;
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_uint64;
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_bfloat16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = Less_init;
		n->exit = Less_exit;
		n->operator = Less_float64;
		break;
	default:
		break;
	}
}
