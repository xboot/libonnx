#include <onnx.h>

void resolver_default_op_TensorScatter(struct onnx_node_t * n)
{
	if(n->opset >= 24)
	{
	}
}
