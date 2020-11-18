#include <onnx.h>

static int Not_init(struct onnx_node_t * n)
{
	if((n->ninput == 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Not_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Not_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, ONNX_TENSOR_TYPE_BOOL);
}

static void Not_bool(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint8_t * px = (uint8_t *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = !px[i];
}

void resolver_default_op_Not(struct onnx_node_t * n)
{
	if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Not_init;
			n->exit = Not_exit;
			n->reshape = Not_reshape;
			n->operator = Not_bool;
			break;
		default:
			break;
		}
	}
}
