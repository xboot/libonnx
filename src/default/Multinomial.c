#include <onnx.h>

struct operator_pdata_t {
	enum onnx_tensor_type_t dtype;
	int sample_size;
	float seed;
};

static int Multinomial_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->dtype = (enum onnx_tensor_type_t)onnx_attribute_read_int(n, "dtype", 6);
			pdat->sample_size = onnx_attribute_read_int(n, "sample_size", 1);
			pdat->seed = onnx_attribute_read_float(n, "seed", 0.0);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int Multinomial_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int Multinomial_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, pdat->dtype);
}

static void Multinomial_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int bsz = x->dims[0];
	int csz = x->dims[1];
	uint16_t * px = (uint16_t *)x->datas;
	float cum[csz];
	int i, j, k, l, o;

	if(pdat->seed != 0.0)
		srand(pdat->seed);

	switch(y->type)
	{
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0; i < bsz; i++)
			{
				for(j = 0; j < pdat->sample_size; j++)
				{
					cum[0] = float16_to_float32(px[i * csz]);
					for(k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + float16_to_float32(px[i * csz + k]);
					for(k = 0, l = csz - 1; k < csz - 1; k++)
					{
						if((float)rand() / (float)(RAND_MAX) < cum[k])
						{
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0; i < bsz; i++)
			{
				for(j = 0; j < pdat->sample_size; j++)
				{
					cum[0] = float16_to_float32(px[i * csz]);
					for(k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + float16_to_float32(px[i * csz + k]);
					for(k = 0, l = csz - 1; k < csz - 1; k++)
					{
						if((float)rand() / (float)(RAND_MAX) < cum[k])
						{
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	default:
		break;
	}
}

static void Multinomial_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int bsz = x->dims[0];
	int csz = x->dims[1];
	float * px = (float *)x->datas;
	float cum[csz];
	int i, j, k, l, o;

	if(pdat->seed != 0.0)
		srand(pdat->seed);

	switch(y->type)
	{
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0; i < bsz; i++)
			{
				for(j = 0; j < pdat->sample_size; j++)
				{
					cum[0] = px[i * csz];
					for(k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + px[i * csz + k];
					for(k = 0, l = csz - 1; k < csz - 1; k++)
					{
						if((float)rand() / (float)(RAND_MAX) < cum[k])
						{
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0; i < bsz; i++)
			{
				for(j = 0; j < pdat->sample_size; j++)
				{
					cum[0] = px[i * csz];
					for(k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + px[i * csz + k];
					for(k = 0, l = csz - 1; k < csz - 1; k++)
					{
						if((float)rand() / (float)(RAND_MAX) < cum[k])
						{
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	default:
		break;
	}
}

static void Multinomial_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int bsz = x->dims[0];
	int csz = x->dims[1];
	double * px = (double *)x->datas;
	double cum[csz];
	int i, j, k, l, o;

	if(pdat->seed != 0.0)
		srand(pdat->seed);

	switch(y->type)
	{
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0; i < bsz; i++)
			{
				for(j = 0; j < pdat->sample_size; j++)
				{
					cum[0] = px[i * csz];
					for(k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + px[i * csz + k];
					for(k = 0, l = csz - 1; k < csz - 1; k++)
					{
						if((double)rand() / (double)(RAND_MAX) < cum[k])
						{
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0; i < bsz; i++)
			{
				for(j = 0; j < pdat->sample_size; j++)
				{
					cum[0] = px[i * csz];
					for(k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + px[i * csz + k];
					for(k = 0, l = csz - 1; k < csz - 1; k++)
					{
						if((double)rand() / (double)(RAND_MAX) < cum[k])
						{
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	default:
		break;
	}
}

void resolver_default_op_Multinomial(struct onnx_node_t * n)
{
	if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Multinomial_init;
			n->exit = Multinomial_exit;
			n->reshape = Multinomial_reshape;
			n->operator = Multinomial_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Multinomial_init;
			n->exit = Multinomial_exit;
			n->reshape = Multinomial_reshape;
			n->operator = Multinomial_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Multinomial_init;
			n->exit = Multinomial_exit;
			n->reshape = Multinomial_reshape;
			n->operator = Multinomial_float64;
			break;
		default:
			break;
		}
	}
}
