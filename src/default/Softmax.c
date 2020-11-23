#include <onnx.h>

struct operator_13_pdata_t
{
	int axis;

	int caxis;
	int current;
	int outter;
	int inner;
};

static int Softmax_13_init(struct onnx_node_t * n)
{
	struct operator_13_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_13_pdata_t));
		if(pdat)
		{
			pdat->axis = onnx_attribute_read_int(n, "axis", -1);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int Softmax_13_exit(struct onnx_node_t * n)
{
	struct operator_13_pdata_t * pdat = (struct operator_13_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int Softmax_13_reshape(struct onnx_node_t * n)
{
	struct operator_13_pdata_t * pdat = (struct operator_13_pdata_t *)n->priv;
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

static void Softmax_13_bfloat16(struct onnx_node_t * n)
{
	struct operator_13_pdata_t * pdat = (struct operator_13_pdata_t *)n->priv;
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
					py[io] = float32_to_bfloat16(v / sum);
				}
			}
		}
	}
}

static void Softmax_13_float16(struct onnx_node_t * n)
{
	struct operator_13_pdata_t * pdat = (struct operator_13_pdata_t *)n->priv;
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
					py[io] = float32_to_float16(v / sum);
				}
			}
		}
	}
}

static void Softmax_13_float32(struct onnx_node_t * n)
{
	struct operator_13_pdata_t * pdat = (struct operator_13_pdata_t *)n->priv;
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
					py[io] /= sum;
				}
			}
		}
	}
}

static void Softmax_13_float64(struct onnx_node_t * n)
{
	struct operator_13_pdata_t * pdat = (struct operator_13_pdata_t *)n->priv;
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
					py[io] /= sum;
				}
			}
		}
	}
}

struct operator_1_11_pdata_t {
	int axis;

	int N;
	int D;
};

static int Softmax_1_11_init(struct onnx_node_t * n)
{
	struct operator_1_11_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_1_11_pdata_t));
		if(pdat)
		{
			pdat->axis = onnx_attribute_read_int(n, "axis", 1);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int Softmax_1_11_exit(struct onnx_node_t * n)
{
	struct operator_1_11_pdata_t * pdat = (struct operator_1_11_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int Softmax_1_11_reshape(struct onnx_node_t * n)
{
	struct operator_1_11_pdata_t * pdat = (struct operator_1_11_pdata_t *)n->priv;
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

static void Softmax_1_11_float16(struct onnx_node_t * n)
{
	struct operator_1_11_pdata_t * pdat = (struct operator_1_11_pdata_t *)n->priv;
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

static void Softmax_1_11_float32(struct onnx_node_t * n)
{
	struct operator_1_11_pdata_t * pdat = (struct operator_1_11_pdata_t *)n->priv;
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

static void Softmax_1_11_float64(struct onnx_node_t * n)
{
	struct operator_1_11_pdata_t * pdat = (struct operator_1_11_pdata_t *)n->priv;
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
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Softmax_13_init;
			n->exit = Softmax_13_exit;
			n->reshape = Softmax_13_reshape;
			n->operator = Softmax_13_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Softmax_13_init;
			n->exit = Softmax_13_exit;
			n->reshape = Softmax_13_reshape;
			n->operator = Softmax_13_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Softmax_13_init;
			n->exit = Softmax_13_exit;
			n->reshape = Softmax_13_reshape;
			n->operator = Softmax_13_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Softmax_13_init;
			n->exit = Softmax_13_exit;
			n->reshape = Softmax_13_reshape;
			n->operator = Softmax_13_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 11)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Softmax_1_11_init;
			n->exit = Softmax_1_11_exit;
			n->reshape = Softmax_1_11_reshape;
			n->operator = Softmax_1_11_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Softmax_1_11_init;
			n->exit = Softmax_1_11_exit;
			n->reshape = Softmax_1_11_reshape;
			n->operator = Softmax_1_11_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Softmax_1_11_init;
			n->exit = Softmax_1_11_exit;
			n->reshape = Softmax_1_11_reshape;
			n->operator = Softmax_1_11_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Softmax_1_11_init;
			n->exit = Softmax_1_11_exit;
			n->reshape = Softmax_1_11_reshape;
			n->operator = Softmax_1_11_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Softmax_1_11_init;
			n->exit = Softmax_1_11_exit;
			n->reshape = Softmax_1_11_reshape;
			n->operator = Softmax_1_11_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Softmax_1_11_init;
			n->exit = Softmax_1_11_exit;
			n->reshape = Softmax_1_11_reshape;
			n->operator = Softmax_1_11_float64;
			break;
		default:
			break;
		}
	}
}
