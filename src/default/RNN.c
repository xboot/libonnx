#include "../onnx.h"

void resolver_default_op_RNN(struct onnx_node_t * n)
{
	if(n->opset >= 14)
	{
	}
	else if(n->opset >= 7)
	{
	}
	else if(n->opset >= 1)
	{
	}
}
