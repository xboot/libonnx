#include <onnx.h>

struct operator_pdata_t {
	float alpha;
	float beta;
};

static int HardSigmoid_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput > 0) && (n->noutput > 0))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
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

static int HardSigmoid_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
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
	if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = HardSigmoid_init;
			n->exit = HardSigmoid_exit;
			n->reshape = HardSigmoid_reshape;
			n->operator = HardSigmoid_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = HardSigmoid_init;
			n->exit = HardSigmoid_exit;
			n->reshape = HardSigmoid_reshape;
			n->operator = HardSigmoid_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = HardSigmoid_init;
			n->exit = HardSigmoid_exit;
			n->reshape = HardSigmoid_reshape;
			n->operator = HardSigmoid_float64;
			break;
		default:
			break;
		}
	}
	if(n->opset >= 1)
	{
	}
}
