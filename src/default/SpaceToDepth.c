#include <onnx.h>

void resolver_default_op_SpaceToDepth(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
	}
	else if(n->opset >= 1)
	{
	}
}
