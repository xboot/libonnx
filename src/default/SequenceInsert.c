#include <onnx.h>

void resolver_default_op_SequenceInsert(struct onnx_node_t * n)
{
	if(n->opset >= 11)
	{
	}
}
