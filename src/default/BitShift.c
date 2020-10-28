#include <onnx.h>

enum bit_shift_direction_t {
	BIT_SHIFT_DIRECTION_LEFT	= 0,
	BIT_SHIFT_DIRECTION_RIGHT	= 1,
};

struct operator_pdata_t {
	enum bit_shift_direction_t dir;
};

static int BitShift_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 2) && (n->noutput == 1))
	{
		struct onnx_tensor_t * a = n->inputs[0];
		struct onnx_tensor_t * b = n->inputs[1];
		struct onnx_tensor_t * y = n->outputs[0];
		if(onnx_tensor_reshape_multi_broadcast(a, b, y, a->type))
		{
			pdat = malloc(sizeof(struct operator_pdata_t));
			if(pdat)
			{
				pdat->dir = (strcmp(onnx_attribute_read_string(n, "direction", "LEFT"), "LEFT") == 0) ? BIT_SHIFT_DIRECTION_LEFT : BIT_SHIFT_DIRECTION_RIGHT;
				n->priv = pdat;
				return 1;
			}
		}
	}
	return 0;
}

static int BitShift_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static void BitShift_uint8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * pa;
	uint8_t * pb;
	int i, l;

	if(onnx_tensor_reshape_multi_broadcast(a, b, y, a->type))
	{
		if(pdat->dir == BIT_SHIFT_DIRECTION_LEFT)
		{
			for(i = 0, l = y->ndata; i < l; i++)
			{
				pa = onnx_tensor_broadcast_map_address(a, y, i);
				pb = onnx_tensor_broadcast_map_address(b, y, i);
				py[i] = *pa << *pb;
			}
		}
		else
		{
			for(i = 0, l = y->ndata; i < l; i++)
			{
				pa = onnx_tensor_broadcast_map_address(a, y, i);
				pb = onnx_tensor_broadcast_map_address(b, y, i);
				py[i] = *pa >> *pb;
			}
		}
	}
}

static void BitShift_uint16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;
	int i, l;

	if(onnx_tensor_reshape_multi_broadcast(a, b, y, a->type))
	{
		if(pdat->dir == BIT_SHIFT_DIRECTION_LEFT)
		{
			for(i = 0, l = y->ndata; i < l; i++)
			{
				pa = onnx_tensor_broadcast_map_address(a, y, i);
				pb = onnx_tensor_broadcast_map_address(b, y, i);
				py[i] = *pa << *pb;
			}
		}
		else
		{
			for(i = 0, l = y->ndata; i < l; i++)
			{
				pa = onnx_tensor_broadcast_map_address(a, y, i);
				pb = onnx_tensor_broadcast_map_address(b, y, i);
				py[i] = *pa >> *pb;
			}
		}
	}
}

static void BitShift_uint32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t * pa;
	uint32_t * pb;
	int i, l;

	if(onnx_tensor_reshape_multi_broadcast(a, b, y, a->type))
	{
		if(pdat->dir == BIT_SHIFT_DIRECTION_LEFT)
		{
			for(i = 0, l = y->ndata; i < l; i++)
			{
				pa = onnx_tensor_broadcast_map_address(a, y, i);
				pb = onnx_tensor_broadcast_map_address(b, y, i);
				py[i] = *pa << *pb;
			}
		}
		else
		{
			for(i = 0, l = y->ndata; i < l; i++)
			{
				pa = onnx_tensor_broadcast_map_address(a, y, i);
				pb = onnx_tensor_broadcast_map_address(b, y, i);
				py[i] = *pa >> *pb;
			}
		}
	}
}

static void BitShift_uint64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * pa;
	uint64_t * pb;
	int i, l;

	if(onnx_tensor_reshape_multi_broadcast(a, b, y, a->type))
	{
		if(pdat->dir == BIT_SHIFT_DIRECTION_LEFT)
		{
			for(i = 0, l = y->ndata; i < l; i++)
			{
				pa = onnx_tensor_broadcast_map_address(a, y, i);
				pb = onnx_tensor_broadcast_map_address(b, y, i);
				py[i] = *pa << *pb;
			}
		}
		else
		{
			for(i = 0, l = y->ndata; i < l; i++)
			{
				pa = onnx_tensor_broadcast_map_address(a, y, i);
				pb = onnx_tensor_broadcast_map_address(b, y, i);
				py[i] = *pa >> *pb;
			}
		}
	}
}

void resolver_default_op_BitShift(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_UINT8:
		n->init = BitShift_init;
		n->exit = BitShift_exit;
		n->operator = BitShift_uint8;
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		n->init = BitShift_init;
		n->exit = BitShift_exit;
		n->operator = BitShift_uint16;
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		n->init = BitShift_init;
		n->exit = BitShift_exit;
		n->operator = BitShift_uint32;
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		n->init = BitShift_init;
		n->exit = BitShift_exit;
		n->operator = BitShift_uint64;
		break;
	default:
		break;
	}
}
