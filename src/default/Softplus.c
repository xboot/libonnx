#include <onnx.h>

static int Softplus_init(struct onnx_node_t * n)
{
	if((n->ninput == 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Softplus_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Softplus_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Softplus_float16(struct onnx_node_t * n)
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
		py[i] = float32_to_float16(logf(expf(v) + 1));
	}
}

static void Softplus_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = logf(expf(px[i]) + 1);
}

static void Softplus_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = log(exp(px[i]) + 1);
}

void resolver_default_op_Softplus(struct onnx_node_t * n)
{
	if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Softplus_init;
			n->exit = Softplus_exit;
			n->reshape = Softplus_reshape;
			n->operator = Softplus_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Softplus_init;
			n->exit = Softplus_exit;
			n->reshape = Softplus_reshape;
			n->operator = Softplus_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Softplus_init;
			n->exit = Softplus_exit;
			n->reshape = Softplus_reshape;
			n->operator = Softplus_float64;
			break;
		default:
			break;
		}
	}
}
