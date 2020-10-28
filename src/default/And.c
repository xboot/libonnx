#include <onnx.h>

static int And_init(struct onnx_node_t * n)
{
	if((n->ninput == 2) && (n->noutput == 1))
	{
		struct onnx_tensor_t * a = n->inputs[0];
		struct onnx_tensor_t * b = n->inputs[1];
		struct onnx_tensor_t * y = n->outputs[0];
		if(onnx_tensor_multi_broadcast_reshape(a, b, y, a->type))
			return 1;
	}
	return 0;
}

static int And_exit(struct onnx_node_t * n)
{
	return 1;
}

static void And_bool(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * pa;
	uint8_t * pb;
	int i, l;

	if(onnx_tensor_multi_broadcast_reshape(a, b, y, a->type))
	{
		for(i = 0, l = y->ndata; i < l; i++)
		{
			pa = onnx_tensor_broadcast_map_address(a, y, i);
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = (*pa && *pb) ? 1 : 0;
		}
	}
}

void resolver_default_op_And(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		n->init = And_init;
		n->exit = And_exit;
		n->operator = And_bool;
		break;
	default:
		break;
	}
}
