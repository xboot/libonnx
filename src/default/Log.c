#include "../onnx.h"

static int Log_init(struct onnx_node_t * n)
{
	if((n->ninput == 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Log_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Log_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Log_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	size_t i,l;
	for(i=0, l = y->ndata; i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		py[i] = float32_to_bfloat16(logf(v));
	}
}

static void Log_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	size_t i,l;
	for(i=0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(logf(v));
	}
}

static void Log_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	size_t i,l;
	for(i=0, l = y->ndata; i < l; i++)
		py[i] = logf(px[i]);
}

static void Log_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	size_t i,l;
	for(i=0, l = y->ndata; i < l; i++)
		py[i] = log(px[i]);
}

void resolver_default_op_Log(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Log_init;
			n->exit = Log_exit;
			n->reshape = Log_reshape;
			n->operator = Log_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Log_init;
			n->exit = Log_exit;
			n->reshape = Log_reshape;
			n->operator = Log_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Log_init;
			n->exit = Log_exit;
			n->reshape = Log_reshape;
			n->operator = Log_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Log_init;
			n->exit = Log_exit;
			n->reshape = Log_reshape;
			n->operator = Log_float64;
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
			n->init = Log_init;
			n->exit = Log_exit;
			n->reshape = Log_reshape;
			n->operator = Log_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Log_init;
			n->exit = Log_exit;
			n->reshape = Log_reshape;
			n->operator = Log_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Log_init;
			n->exit = Log_exit;
			n->reshape = Log_reshape;
			n->operator = Log_float64;
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
			n->init = Log_init;
			n->exit = Log_exit;
			n->reshape = Log_reshape;
			n->operator = Log_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Log_init;
			n->exit = Log_exit;
			n->reshape = Log_reshape;
			n->operator = Log_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Log_init;
			n->exit = Log_exit;
			n->reshape = Log_reshape;
			n->operator = Log_float64;
			break;
		default:
			break;
		}
	}
}
