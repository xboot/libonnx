#include "../onnx.h"

void resolver_default_op_Scatter(struct onnx_node_t * n)
{
	if(n->opset >= 11)
	{
		return;
	}
	else if(n->opset >= 9)
	{
	}
}
