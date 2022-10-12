#include "../onnx.h"

static int Dropout_init(struct onnx_node_t * n)
{
	if((n->ninput >= 1) && (n->noutput >= 1))
		return 1;
	return 0;
}

static int Dropout_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Dropout_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Dropout_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;

	size_t i,l;
	for(i=0, l = y->ndata; i < l; i++)
		py[i] = px[i];
}

static void Dropout_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;

	size_t i,l;
	for(i=0, l = y->ndata; i < l; i++)
		py[i] = px[i];
}

static void Dropout_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	size_t i,l;
	for(i=0, l = y->ndata; i < l; i++)
		py[i] = px[i];
}

static void Dropout_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	size_t i,l;
	for(i=0, l = y->ndata; i < l; i++)
		py[i] = px[i];
}

void resolver_default_op_Dropout(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 12)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 10)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float64;
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
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float64;
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
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Dropout_init;
			n->exit = Dropout_exit;
			n->reshape = Dropout_reshape;
			n->operator = Dropout_float64;
			break;
		default:
			break;
		}
	}
}
