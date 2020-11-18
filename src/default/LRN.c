#include <onnx.h>

struct operator_pdata_t {
	float alpha;
	float beta;
	float bias;
	int size;
};

static int LRN_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->alpha = onnx_attribute_read_float(n, "alpha", 0.0001);
			pdat->beta = onnx_attribute_read_float(n, "beta", 0.75);
			pdat->bias = onnx_attribute_read_float(n, "bias", 1.0);
			pdat->size = onnx_attribute_read_int(n, "size", 1);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int LRN_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int LRN_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void LRN_bfloat16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float sum, t;
	float over = pdat->alpha / pdat->size;
	int N = x->dims[0];
	int C = x->dims[1];
	int L = x->strides[1];
	int start, end;
	int i, j, u, v, o;

	for(u = 0; u < N; u++)
	{
		for(v = 0; v < C; v++)
		{
			for(i = 0; i < L; i++)
			{
				start = v - (pdat->size / 2);
				if(start < 0)
					start = 0;
				end = v + (pdat->size / 2);
				if(end >= C)
					end = C - 1;
				for(j = start, sum = 0; j <= end; ++j)
				{
					t = bfloat16_to_float32(px[(u * C + j) * L + i]);
					sum += t * t;
				}
				o = (u * C + v) * L + i;
				py[o] = float32_to_bfloat16(bfloat16_to_float32(px[o]) * powf(pdat->bias + over * sum, -pdat->beta));
			}
		}
	}
}

static void LRN_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float sum, t;
	float over = pdat->alpha / pdat->size;
	int N = x->dims[0];
	int C = x->dims[1];
	int L = x->strides[1];
	int start, end;
	int i, j, u, v, o;

	for(u = 0; u < N; u++)
	{
		for(v = 0; v < C; v++)
		{
			for(i = 0; i < L; i++)
			{
				start = v - (pdat->size / 2);
				if(start < 0)
					start = 0;
				end = v + (pdat->size / 2);
				if(end >= C)
					end = C - 1;
				for(j = start, sum = 0; j <= end; ++j)
				{
					t = float16_to_float32(px[(u * C + j) * L + i]);
					sum += t * t;
				}
				o = (u * C + v) * L + i;
				py[o] = float32_to_float16(float16_to_float32(px[o]) * powf(pdat->bias + over * sum, -pdat->beta));
			}
		}
	}
}

static void LRN_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	float sum, t;
	float over = pdat->alpha / pdat->size;
	int N = x->dims[0];
	int C = x->dims[1];
	int L = x->strides[1];
	int start, end;
	int i, j, u, v, o;

	for(u = 0; u < N; u++)
	{
		for(v = 0; v < C; v++)
		{
			for(i = 0; i < L; i++)
			{
				start = v - (pdat->size / 2);
				if(start < 0)
					start = 0;
				end = v + (pdat->size / 2);
				if(end >= C)
					end = C - 1;
				for(j = start, sum = 0; j <= end; ++j)
				{
					t = px[(u * C + j) * L + i];
					sum += t * t;
				}
				o = (u * C + v) * L + i;
				py[o] = px[o] * powf(pdat->bias + over * sum, -pdat->beta);
			}
		}
	}
}

static void LRN_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	double sum, t;
	double over = pdat->alpha / pdat->size;
	int N = x->dims[0];
	int C = x->dims[1];
	int L = x->strides[1];
	int start, end;
	int i, j, u, v, o;

	for(u = 0; u < N; u++)
	{
		for(v = 0; v < C; v++)
		{
			for(i = 0; i < L; i++)
			{
				start = v - (pdat->size / 2);
				if(start < 0)
					start = 0;
				end = v + (pdat->size / 2);
				if(end >= C)
					end = C - 1;
				for(j = start, sum = 0; j <= end; ++j)
				{
					t = px[(u * C + j) * L + i];
					sum += t * t;
				}
				o = (u * C + v) * L + i;
				py[o] = px[o] * pow(pdat->bias + over * sum, -pdat->beta);
			}
		}
	}
}

void resolver_default_op_LRN(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = LRN_init;
			n->exit = LRN_exit;
			n->reshape = LRN_reshape;
			n->operator = LRN_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = LRN_init;
			n->exit = LRN_exit;
			n->reshape = LRN_reshape;
			n->operator = LRN_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = LRN_init;
			n->exit = LRN_exit;
			n->reshape = LRN_reshape;
			n->operator = LRN_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = LRN_init;
			n->exit = LRN_exit;
			n->reshape = LRN_reshape;
			n->operator = LRN_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
	}
}
