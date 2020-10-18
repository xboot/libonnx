#include <onnx.h>

struct operator_pdata_t {
	int detect_negative;
	int detect_positive;
};

static void IsInf_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	struct onnx_tensor_t * t = n->inputs[0];
	int i;

	for(i = 0; i < n->noutput; i++)
	{
		if(n->outputs[i]->type == ONNX_TENSOR_TYPE_UNDEFINED)
			onnx_tensor_reinit(n->outputs[i], ONNX_TENSOR_TYPE_BOOL, t->dims, t->ndim);
	}

	pdat = malloc(sizeof(struct operator_pdata_t));
	if(pdat)
	{
		pdat->detect_negative = onnx_attribute_read_int(n, "detect_negative", 1);
		pdat->detect_positive = onnx_attribute_read_int(n, "detect_positive", 1);
	}
	n->priv = pdat;
}

static void IsInf_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
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
		switch(isinff(px[i]))
		{
		case -1:
			if(pdat->detect_negative)
				py[i] = 1;
			else
				py[i] = 0;
			break;
		case 0:
			py[i] = 0;
			break;
		case 1:
			if(pdat->detect_positive)
				py[i] = 1;
			else
				py[i] = 0;
			break;
		default:
			break;
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
		switch(isinf(px[i]))
		{
		case -1:
			if(pdat->detect_negative)
				py[i] = 1;
			else
				py[i] = 0;
			break;
		case 0:
			py[i] = 0;
			break;
		case 1:
			if(pdat->detect_positive)
				py[i] = 1;
			else
				py[i] = 0;
			break;
		default:
			break;
		}
	}
}

void default_resolver_op_IsInf(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = IsInf_init;
		n->exit = IsInf_exit;
		n->op = IsInf_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = IsInf_init;
		n->exit = IsInf_exit;
		n->op = IsInf_float64;
		break;
	default:
		break;
	}
}
