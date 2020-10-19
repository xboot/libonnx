#include <onnx.h>

struct operator_pdata_t {
	float alpha;
	float gamma;
};

static void Selu_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	struct onnx_tensor_t * t = n->inputs[0];
	int i;

	for(i = 0; i < n->noutput; i++)
	{
		if(n->outputs[i]->type == ONNX_TENSOR_TYPE_UNDEFINED)
			onnx_tensor_reinit(n->outputs[i], t->type, t->dims, t->ndim);
	}

	pdat = malloc(sizeof(struct operator_pdata_t));
	if(pdat)
	{
		pdat->alpha = onnx_attribute_read_float(n, "alpha", 1.67326);
		pdat->gamma = onnx_attribute_read_float(n, "gamma", 1.0507);
	}
	n->priv = pdat;
}

static void Selu_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
}

static void Selu_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		if(v > 0)
			py[i] = pdat->gamma * v;
		else
			py[i] = pdat->gamma * (pdat->alpha * expf(v) - pdat->alpha);
	}
}

static void Selu_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] > 0)
			py[i] = pdat->gamma * px[i];
		else
			py[i] = pdat->gamma * (pdat->alpha * expf(px[i]) - pdat->alpha);
	}
}

static void Selu_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] > 0)
			py[i] = pdat->gamma * px[i];
		else
			py[i] = pdat->gamma * (pdat->alpha * exp(px[i]) - pdat->alpha);
	}
}

void default_resolver_op_Selu(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = Selu_init;
		n->exit = Selu_exit;
		n->op = Selu_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = Selu_init;
		n->exit = Selu_exit;
		n->op = Selu_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = Selu_init;
		n->exit = Selu_exit;
		n->op = Selu_float64;
		break;
	default:
		break;
	}
}
