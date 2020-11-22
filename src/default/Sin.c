#include <onnx.h>

static int Sin_init(struct onnx_node_t * n)
{
	if((n->ninput == 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Sin_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Sin_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Sin_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(sinf(v));
	}
}

static void Sin_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = sinf(px[i]);
}

static void Sin_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = sin(px[i]);
}

void resolver_default_op_Sin(struct onnx_node_t * n)
{
	if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Sin_init;
			n->exit = Sin_exit;
			n->reshape = Sin_reshape;
			n->operator = Sin_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Sin_init;
			n->exit = Sin_exit;
			n->reshape = Sin_reshape;
			n->operator = Sin_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Sin_init;
			n->exit = Sin_exit;
			n->reshape = Sin_reshape;
			n->operator = Sin_float64;
			break;
		default:
			break;
		}
	}
}
