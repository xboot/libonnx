#include <onnx.h>

void resolver_default_op_Upsample(struct onnx_node_t * n)
{
	if(n->opset >= 10)
	{
		return;
	}
	else if(n->opset >= 9)
	{
	}
	else if(n->opset >= 7)
	{
	}
}
