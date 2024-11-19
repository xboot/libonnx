#include <onnx.h>

static int BitwiseAnd_init(struct onnx_node_t * n)
{
	if((n->ninput == 2) && (n->noutput == 1))
		return 1;
	return 0;
}

static int BitwiseAnd_exit(struct onnx_node_t * n)
{
	return 1;
}

static int BitwiseAnd_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];

	return onnx_tensor_reshape_multi_broadcast(y, a, b, a->type);
}

static void BitwiseAnd_int8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	int8_t * py = (int8_t *)y->datas;
	int8_t * pa;
	int8_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = *pa & *pb;
	}
}

static void BitwiseAnd_int16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	int16_t * py = (int16_t *)y->datas;
	int16_t * pa;
	int16_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = *pa & *pb;
	}
}

static void BitwiseAnd_int32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	int32_t * py = (int32_t *)y->datas;
	int32_t * pa;
	int32_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = *pa & *pb;
	}
}

static void BitwiseAnd_int64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	int64_t * py = (int64_t *)y->datas;
	int64_t * pa;
	int64_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = *pa & *pb;
	}
}

static void BitwiseAnd_uint8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * pa;
	uint8_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = *pa & *pb;
	}
}

static void BitwiseAnd_uint16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = *pa & *pb;
	}
}

static void BitwiseAnd_uint32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t * pa;
	uint32_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = *pa & *pb;
	}
}

static void BitwiseAnd_uint64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * pa;
	uint64_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = *pa & *pb;
	}
}

void resolver_default_op_BitwiseAnd(struct onnx_node_t * n)
{
	if(n->opset >= 18)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = BitwiseAnd_init;
			n->exit = BitwiseAnd_exit;
			n->reshape = BitwiseAnd_reshape;
			n->operator = BitwiseAnd_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = BitwiseAnd_init;
			n->exit = BitwiseAnd_exit;
			n->reshape = BitwiseAnd_reshape;
			n->operator = BitwiseAnd_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = BitwiseAnd_init;
			n->exit = BitwiseAnd_exit;
			n->reshape = BitwiseAnd_reshape;
			n->operator = BitwiseAnd_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = BitwiseAnd_init;
			n->exit = BitwiseAnd_exit;
			n->reshape = BitwiseAnd_reshape;
			n->operator = BitwiseAnd_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = BitwiseAnd_init;
			n->exit = BitwiseAnd_exit;
			n->reshape = BitwiseAnd_reshape;
			n->operator = BitwiseAnd_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = BitwiseAnd_init;
			n->exit = BitwiseAnd_exit;
			n->reshape = BitwiseAnd_reshape;
			n->operator = BitwiseAnd_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = BitwiseAnd_init;
			n->exit = BitwiseAnd_exit;
			n->reshape = BitwiseAnd_reshape;
			n->operator = BitwiseAnd_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = BitwiseAnd_init;
			n->exit = BitwiseAnd_exit;
			n->reshape = BitwiseAnd_reshape;
			n->operator = BitwiseAnd_uint64;
			break;
		default:
			break;
		}
	}
}
