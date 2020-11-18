#include <onnx.h>

void resolver_default_op_Scan(struct onnx_node_t * n)
{
	if(n->opset >= 11)
	{
	}
	else if(n->opset >= 9)
	{
	}
	else if(n->opset >= 8)
	{
	}
}
