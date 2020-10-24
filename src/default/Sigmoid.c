#include <onnx.h>

static int Sigmoid_init(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x;
	struct onnx_tensor_t * y;

	if((n->ninput > 0) && (n->noutput > 0))
	{
		x = n->inputs[0];
		y = n->outputs[0];
		if(!onnx_tensor_shape_equal(y, x) || (y->type != x->type))
			onnx_tensor_reinit(y, x->type, x->dims, x->ndim);
		return 1;
	}
	return 0;
}

static int Sigmoid_exit(struct onnx_node_t * n)
{
	return 1;
}

static void Sigmoid_bfloat16(struct onnx_node_t * n)
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
		if(v >= 0)
			py[i] = float32_to_bfloat16(1.0 / (1.0 + expf(-1 * v)));
		else
			py[i] = float32_to_bfloat16(expf(v) / (1.0 + expf(v)));
	}
}

static void Sigmoid_float16(struct onnx_node_t * n)
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
		if(v >= 0)
			py[i] = float32_to_float16(1.0 / (1.0 + expf(-1 * v)));
		else
			py[i] = float32_to_float16(expf(v) / (1.0 + expf(v)));
	}
}

static void Sigmoid_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] >= 0)
			py[i] = 1.0 / (1.0 + expf(-1 * px[i]));
		else
			py[i] = expf(px[i]) / (1.0 + expf(px[i]));
	}
}

static void Sigmoid_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] >= 0)
			py[i] = 1.0 / (1.0 + exp(-1 * px[i]));
		else
			py[i] = exp(px[i]) / (1.0 + exp(px[i]));
	}
}

void resolver_default_op_Sigmoid(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_BFLOAT16:
		n->init = Sigmoid_init;
		n->exit = Sigmoid_exit;
		n->op = Sigmoid_bfloat16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = Sigmoid_init;
		n->exit = Sigmoid_exit;
		n->op = Sigmoid_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = Sigmoid_init;
		n->exit = Sigmoid_exit;
		n->op = Sigmoid_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = Sigmoid_init;
		n->exit = Sigmoid_exit;
		n->op = Sigmoid_float64;
		break;
	default:
		break;
	}
}
