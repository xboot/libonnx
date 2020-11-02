#include <onnx.h>

struct operator_pdata_t
{
	int axis;

	int caxis;
	int current;
	int outter;
	int inner;
};

static int LogSoftmax_init(struct onnx_node_t * n)
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

static int LogSoftmax_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int LogSoftmax_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int i;

	pdat->caxis = pdat->axis;
	if(pdat->caxis < 0)
		pdat->caxis += x->ndim;
	if(pdat->caxis < 0 || pdat->caxis >= x->ndim)
		return 0;
	for(i = 0, pdat->outter = 1, pdat->inner = 1; i < x->ndim; i++)
	{
		if(i == pdat->caxis)
			pdat->current = x->dims[i];
		else if(i < pdat->caxis)
			pdat->outter *= x->dims[i];
		else
			pdat->inner *= x->dims[i];
	}
	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void LogSoftmax_bfloat16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float maxv, sum, v;
	int i, j, k, o, oo, io;

	for(i = 0; i < pdat->outter; i++)
	{
		oo = i * pdat->current * pdat->inner;
		for(k = 0; k < pdat->inner; k++)
		{
			io = oo + k;
			for(j = 0, maxv = bfloat16_to_float32(px[io]); j < pdat->current; j++)
			{
				o = io + j * pdat->inner;
				v = bfloat16_to_float32(px[o]);
				if(v > maxv)
					maxv = v;
			}
			for(j = 0, sum = 0; j < pdat->current; j++)
			{
				o = io + j * pdat->inner;
				v = expf(bfloat16_to_float32(px[o]) - maxv);
				py[o] = float32_to_bfloat16(v);
				sum += v;
			}
			if(sum != 0)
			{
				for(j = 0; j < pdat->current; j++)
				{
					io = oo + j * pdat->inner + k;
					v = bfloat16_to_float32(py[io]);
					py[io] = float32_to_bfloat16(logf(v / sum));
				}
			}
		}
	}
}

static void LogSoftmax_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float maxv, sum, v;
	int i, j, k, o, oo, io;

	for(i = 0; i < pdat->outter; i++)
	{
		oo = i * pdat->current * pdat->inner;
		for(k = 0; k < pdat->inner; k++)
		{
			io = oo + k;
			for(j = 0, maxv = float16_to_float32(px[io]); j < pdat->current; j++)
			{
				o = io + j * pdat->inner;
				v = float16_to_float32(px[o]);
				if(v > maxv)
					maxv = v;
			}
			for(j = 0, sum = 0; j < pdat->current; j++)
			{
				o = io + j * pdat->inner;
				v = expf(float16_to_float32(px[o]) - maxv);
				py[o] = float32_to_float16(v);
				sum += v;
			}
			if(sum != 0)
			{
				for(j = 0; j < pdat->current; j++)
				{
					io = oo + j * pdat->inner + k;
					v = float16_to_float32(py[io]);
					py[io] = float32_to_float16(logf(v / sum));
				}
			}
		}
	}
}

static void LogSoftmax_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	float maxv, sum;
	int i, j, k, o, oo, io;

	for(i = 0; i < pdat->outter; i++)
	{
		oo = i * pdat->current * pdat->inner;
		for(k = 0; k < pdat->inner; k++)
		{
			io = oo + k;
			for(j = 0, maxv = px[io]; j < pdat->current; j++)
			{
				o = io + j * pdat->inner;
				if(px[o] > maxv)
					maxv = px[o];
			}
			for(j = 0, sum = 0; j < pdat->current; j++)
			{
				o = io + j * pdat->inner;
				py[o] = expf(px[o] - maxv);
				sum += py[o];
			}
			if(sum != 0)
			{
				for(j = 0; j < pdat->current; j++)
				{
					io = oo + j * pdat->inner + k;
					py[io] = logf(py[io] / sum);
				}
			}
		}
	}
}

static void LogSoftmax_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	double maxv, sum;
	int i, j, k, o, oo, io;

	for(i = 0; i < pdat->outter; i++)
	{
		oo = i * pdat->current * pdat->inner;
		for(k = 0; k < pdat->inner; k++)
		{
			io = oo + k;
			for(j = 0, maxv = px[io]; j < pdat->current; j++)
			{
				o = io + j * pdat->inner;
				if(px[o] > maxv)
					maxv = px[o];
			}
			for(j = 0, sum = 0; j < pdat->current; j++)
			{
				o = io + j * pdat->inner;
				py[o] = exp(px[o] - maxv);
				sum += py[o];
			}
			if(sum != 0)
			{
				for(j = 0; j < pdat->current; j++)
				{
					io = oo + j * pdat->inner + k;
					py[io] = log(py[io] / sum);
				}
			}
		}
	}
}

void resolver_default_op_LogSoftmax(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_BFLOAT16:
		n->init = LogSoftmax_init;
		n->exit = LogSoftmax_exit;
		n->reshape = LogSoftmax_reshape;
		n->operator = LogSoftmax_bfloat16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = LogSoftmax_init;
		n->exit = LogSoftmax_exit;
		n->reshape = LogSoftmax_reshape;
		n->operator = LogSoftmax_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = LogSoftmax_init;
		n->exit = LogSoftmax_exit;
		n->reshape = LogSoftmax_reshape;
		n->operator = LogSoftmax_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = LogSoftmax_init;
		n->exit = LogSoftmax_exit;
		n->reshape = LogSoftmax_reshape;
		n->operator = LogSoftmax_float64;
		break;
	default:
		break;
	}
}
