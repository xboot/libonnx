#include <onnx.h>

struct operator_pdata_t {
	float alpha;
	float beta;
};

static int HardSigmoid_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	struct onnx_tensor_t * x;
	struct onnx_tensor_t * y;

	if((n->ninput > 0) && (n->noutput > 0))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			x = n->inputs[0];
			y = n->outputs[0];
			if(!onnx_tensor_shape_equal(y, x) || (y->type != x->type))
				onnx_tensor_reinit(y, x->type, x->dims, x->ndim);
			pdat->alpha = onnx_attribute_read_float(n, "alpha", 0.2);
			pdat->beta = onnx_attribute_read_float(n, "beta", 0.5);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int HardSigmoid_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static void HardSigmoid_float16(struct onnx_node_t * n)
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
		py[i] = float32_to_float16(max((float)0.0, min((float)1.0, (float)(pdat->alpha * v + pdat->beta))));
	}
}

static void HardSigmoid_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = max((float)0.0, min((float)1.0, (float)(pdat->alpha * px[i] + pdat->beta)));
}

static void HardSigmoid_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = max((double)0.0, min((double)1.0, (double)(pdat->alpha * px[i] + pdat->beta)));
}

void resolver_default_op_HardSigmoid(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = HardSigmoid_init;
		n->exit = HardSigmoid_exit;
		n->op = HardSigmoid_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = HardSigmoid_init;
		n->exit = HardSigmoid_exit;
		n->op = HardSigmoid_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = HardSigmoid_init;
		n->exit = HardSigmoid_exit;
		n->op = HardSigmoid_float64;
		break;
	default:
		break;
	}
}
