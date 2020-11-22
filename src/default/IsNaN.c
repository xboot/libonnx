#include <onnx.h>

static int IsNaN_init(struct onnx_node_t * n)
{
	if((n->ninput == 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int IsNaN_exit(struct onnx_node_t * n)
{
	return 1;
}

static int IsNaN_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, ONNX_TENSOR_TYPE_BOOL);
}

static void IsNaN_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		py[i] = isnanf(v) ? 1 : 0;
	}
}

static void IsNaN_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = isnanf(v) ? 1 : 0;
	}
}

static void IsNaN_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = isnanf(px[i]) ? 1 : 0;
}

static void IsNaN_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = isnan(px[i]) ? 1 : 0;
}

void resolver_default_op_IsNaN(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = IsNaN_init;
			n->exit = IsNaN_exit;
			n->reshape = IsNaN_reshape;
			n->operator = IsNaN_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = IsNaN_init;
			n->exit = IsNaN_exit;
			n->reshape = IsNaN_reshape;
			n->operator = IsNaN_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = IsNaN_init;
			n->exit = IsNaN_exit;
			n->reshape = IsNaN_reshape;
			n->operator = IsNaN_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = IsNaN_init;
			n->exit = IsNaN_exit;
			n->reshape = IsNaN_reshape;
			n->operator = IsNaN_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 9)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = IsNaN_init;
			n->exit = IsNaN_exit;
			n->reshape = IsNaN_reshape;
			n->operator = IsNaN_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = IsNaN_init;
			n->exit = IsNaN_exit;
			n->reshape = IsNaN_reshape;
			n->operator = IsNaN_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = IsNaN_init;
			n->exit = IsNaN_exit;
			n->reshape = IsNaN_reshape;
			n->operator = IsNaN_float64;
			break;
		default:
			break;
		}
	}
}
