#include "../onnx.h"

void resolver_default_op_DepthToSpace(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
	}
	else if(n->opset >= 11)
	{
	}
	else if(n->opset >= 1)
	{
	}
}
