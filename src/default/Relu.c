#include <onnx.h>

static int Relu_init(struct onnx_node_t * n)
{
	if((n->ninput == 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Relu_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Relu_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Relu_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		if(v < 0)
			v = 0;
		py[i] = float32_to_bfloat16(v);
	}
}

static void Relu_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		if(v < 0)
			v = 0;
		py[i] = float32_to_float16(v);
	}
}

static void Relu_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? 0 : px[i];
}

static void Relu_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? 0 : px[i];
}

void resolver_default_op_Relu(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->operator = Relu_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->operator = Relu_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->operator = Relu_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->operator = Relu_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->operator = Relu_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->operator = Relu_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->operator = Relu_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->operator = Relu_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->operator = Relu_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->operator = Relu_float64;
			break;
		default:
			break;
		}
	}
}
