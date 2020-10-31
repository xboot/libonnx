#include <onnx.h>

struct operator_pdata_t {
	int axis;

	int N;
	int D;
};

static int Softmax_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->axis = onnx_attribute_read_int(n, "axis", -1);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int Softmax_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int Softmax_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int axis = pdat->axis;
	int i;

	if(axis < 0)
		axis += x->ndim;
	if(axis < 0 || axis >= x->ndim)
		return 0;
	for(i = 0, pdat->N = 1, pdat->D = 1; i < x->ndim; i++)
	{
		if(i < axis)
			pdat->N *= x->dims[i];
		else
			pdat->D *= x->dims[i];
	}
	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Softmax_bfloat16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float maxv, sum, v;
	int i, j, o;

	for(i = 0, o = 0; i < pdat->N; i++, o += pdat->D)
	{
		for(j = 0, maxv = FLT_MIN; j < pdat->D; j++)
		{
			v = bfloat16_to_float32(px[o + j]);
			if(v > maxv)
				maxv = v;
		}
		for(j = 0, sum = 0; j < pdat->D; j++)
		{
			v = expf(bfloat16_to_float32(px[o + j]) - maxv);
			py[o + j] = float32_to_bfloat16(v);
			sum += v;
		}
		if(sum != 0)
		{
			for(j = 0; j < pdat->D; j++)
			{
				v = bfloat16_to_float32(py[o + j]);
				py[o + j] = float32_to_bfloat16(v / sum);
			}
		}
	}
}

static void Softmax_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float maxv, sum, v;
	int i, j, o;

	for(i = 0, o = 0; i < pdat->N; i++, o += pdat->D)
	{
		for(j = 0, maxv = FLT_MIN; j < pdat->D; j++)
		{
			v = float16_to_float32(px[o + j]);
			if(v > maxv)
				maxv = v;
		}
		for(j = 0, sum = 0; j < pdat->D; j++)
		{
			v = expf(float16_to_float32(px[o + j]) - maxv);
			py[o + j] = float32_to_float16(v);
			sum += v;
		}
		if(sum != 0)
		{
			for(j = 0; j < pdat->D; j++)
			{
				v = float16_to_float32(py[o + j]);
				py[o + j] = float32_to_float16(v / sum);
			}
		}
	}
}

static void Softmax_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	float maxv, sum;
	int i, j, o;

	for(i = 0, o = 0; i < pdat->N; i++, o += pdat->D)
	{
		for(j = 0, maxv = FLT_MIN; j < pdat->D; j++)
		{
			if(px[o + j] > maxv)
				maxv = px[o + j];
		}
		for(j = 0, sum = 0; j < pdat->D; j++)
		{
			py[o + j] = expf(px[o + j] - maxv);
			sum += py[o + j];
		}
		if(sum != 0)
		{
			for(j = 0; j < pdat->D; j++)
				py[o + j] /= sum;
		}
	}
}

static void Softmax_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	double maxv, sum;
	int i, j, o;

	for(i = 0, o = 0; i < pdat->N; i++, o += pdat->D)
	{
		for(j = 0, maxv = DBL_MIN; j < pdat->D; j++)
		{
			if(px[o + j] > maxv)
				maxv = px[o + j];
		}
		for(j = 0, sum = 0; j < pdat->D; j++)
		{
			py[o + j] = exp(px[o + j] - maxv);
			sum += py[o + j];
		}
		if(sum != 0)
		{
			for(j = 0; j < pdat->D; j++)
				py[o + j] /= sum;
		}
	}
}

void resolver_default_op_Softmax(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_BFLOAT16:
		n->init = Softmax_init;
		n->exit = Softmax_exit;
		n->reshape = Softmax_reshape;
		n->operator = Softmax_bfloat16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = Softmax_init;
		n->exit = Softmax_exit;
		n->reshape = Softmax_reshape;
		n->operator = Softmax_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = Softmax_init;
		n->exit = Softmax_exit;
		n->reshape = Softmax_reshape;
		n->operator = Softmax_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = Softmax_init;
		n->exit = Softmax_exit;
		n->reshape = Softmax_reshape;
		n->operator = Softmax_float64;
		break;
	default:
		break;
	}
}
