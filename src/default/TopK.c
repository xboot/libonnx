#include <onnx.h>

void resolver_default_op_TopK(struct onnx_node_t * n)
{
	if(n->opset >= 11)
	{
	}
	else if(n->opset >= 10)
	{
	}
	else if(n->opset >= 1)
	{
	}
}
