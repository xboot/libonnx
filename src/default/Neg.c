#include <onnx.h>

static int Neg_init(struct onnx_node_t * n)
{
	if((n->ninput == 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Neg_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Neg_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Neg_int8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int8_t * px = (int8_t *)x->datas;
	int8_t * py = (int8_t *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = -px[i];
}

static void Neg_int16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int16_t * px = (int16_t *)x->datas;
	int16_t * py = (int16_t *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = -px[i];
}

static void Neg_int32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int32_t * px = (int32_t *)x->datas;
	int32_t * py = (int32_t *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = -px[i];
}

static void Neg_int64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int64_t * px = (int64_t *)x->datas;
	int64_t * py = (int64_t *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = -px[i];
}

static void Neg_bfloat16(struct onnx_node_t * n)
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
		py[i] = float32_to_bfloat16(-v);
	}
}

static void Neg_float16(struct onnx_node_t * n)
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
		py[i] = float32_to_float16(-v);
	}
}

static void Neg_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = -px[i];
}

static void Neg_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = -px[i];
}

void resolver_default_op_Neg(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Neg_init;
			n->exit = Neg_exit;
			n->reshape = Neg_reshape;
			n->operator = Neg_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Neg_init;
			n->exit = Neg_exit;
			n->reshape = Neg_reshape;
			n->operator = Neg_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Neg_init;
			n->exit = Neg_exit;
			n->reshape = Neg_reshape;
			n->operator = Neg_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Neg_init;
			n->exit = Neg_exit;
			n->reshape = Neg_reshape;
			n->operator = Neg_int64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Neg_init;
			n->exit = Neg_exit;
			n->reshape = Neg_reshape;
			n->operator = Neg_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Neg_init;
			n->exit = Neg_exit;
			n->reshape = Neg_reshape;
			n->operator = Neg_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Neg_init;
			n->exit = Neg_exit;
			n->reshape = Neg_reshape;
			n->operator = Neg_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Neg_init;
			n->exit = Neg_exit;
			n->reshape = Neg_reshape;
			n->operator = Neg_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 6)
	{
	}
	else if(n->opset >= 1)
	{
	}
}
