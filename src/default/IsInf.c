#include <onnx.h>

struct operator_pdata_t {
	int detect_negative;
	int detect_positive;
};

static int IsInf_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->detect_negative = onnx_attribute_read_int(n, "detect_negative", 1);
			pdat->detect_positive = onnx_attribute_read_int(n, "detect_positive", 1);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int IsInf_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int IsInf_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, ONNX_TENSOR_TYPE_BOOL);
}

static void IsInf_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(isinff(px[i]))
		{
			if((pdat->detect_negative && (px[i] < 0)) || (pdat->detect_positive && (px[i] > 0)))
				py[i] = 1;
			else
				py[i] = 0;
		}
		else
		{
			py[i] = 0;
		}
	}
}

static void IsInf_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(isinf(px[i]))
		{
			if((pdat->detect_negative && (px[i] < 0)) || (pdat->detect_positive && (px[i] > 0)))
				py[i] = 1;
			else
				py[i] = 0;
		}
		else
		{
			py[i] = 0;
		}
	}
}

void resolver_default_op_IsInf(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = IsInf_init;
		n->exit = IsInf_exit;
		n->reshape = IsInf_reshape;
		n->operator = IsInf_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = IsInf_init;
		n->exit = IsInf_exit;
		n->reshape = IsInf_reshape;
		n->operator = IsInf_float64;
		break;
	default:
		break;
	}
}
