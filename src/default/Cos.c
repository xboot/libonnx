#include <onnx.h>

static void Cos_init(struct onnx_node_t * n)
{
	struct onnx_tensor_t * t = n->inputs[0];
	int i;

	for(i = 0; i < n->noutput; i++)
	{
		if(n->outputs[i]->type == ONNX_TENSOR_TYPE_UNDEFINED)
			onnx_tensor_reinit(n->outputs[i], t->type, t->dims, t->ndim);
	}
}

static void Cos_exit(struct onnx_node_t * n)
{
}

static void Cos_float16(struct onnx_node_t * n)
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
		py[i] = float32_to_float16(cosf(v));
	}
}

static void Cos_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = cosf(px[i]);
}

static void Cos_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = cos(px[i]);
}

void default_resolver_op_Cos(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = Cos_init;
		n->exit = Cos_exit;
		n->op = Cos_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = Cos_init;
		n->exit = Cos_exit;
		n->op = Cos_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = Cos_init;
		n->exit = Cos_exit;
		n->op = Cos_float64;
		break;
	default:
		break;
	}
}
