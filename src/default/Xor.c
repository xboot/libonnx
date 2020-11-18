#include <onnx.h>

static int Xor_init(struct onnx_node_t * n)
{
	if((n->ninput == 2) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Xor_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Xor_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];

	return onnx_tensor_reshape_multi_broadcast(y, a, b, ONNX_TENSOR_TYPE_BOOL);
}

static void Xor_bool(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * pa;
	uint8_t * pb;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa != *pb) ? 1 : 0;
	}
}

void resolver_default_op_Xor(struct onnx_node_t * n)
{
	if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Xor_init;
			n->exit = Xor_exit;
			n->reshape = Xor_reshape;
			n->operator = Xor_bool;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
	}
}
