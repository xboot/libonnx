#include "../onnx.h"

struct operator_pdata_t {
	enum onnx_tensor_type_t dtype;
	float mean;
	float scale;
	float seed;
};

static int RandomNormalLike_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->dtype = (enum onnx_tensor_type_t)onnx_attribute_read_int(n, "dtype", 0);
			pdat->mean = onnx_attribute_read_float(n, "mean", 0.0);
			pdat->scale = onnx_attribute_read_float(n, "scale", 1.0);
			pdat->seed = onnx_attribute_read_float(n, "seed", 0.0);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int RandomNormalLike_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int RandomNormalLike_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	enum onnx_tensor_type_t type;

	if(pdat->dtype != ONNX_TENSOR_TYPE_UNDEFINED)
		type = pdat->dtype;
	else
		type = x->type;
	switch(type)
	{
	case ONNX_TENSOR_TYPE_FLOAT16:
	case ONNX_TENSOR_TYPE_FLOAT32:
	case ONNX_TENSOR_TYPE_FLOAT64:
		return onnx_tensor_reshape(y, x->dims, x->ndim, type);
	default:
		break;
	}
	return 0;
}

static void RandomNormalLike_operator(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];

	if(pdat->seed != 0.0)
		srand(pdat->seed);

	size_t i, l;

	switch(pdat->dtype)
	{
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			float ty, tx;

			for(i = 0, l = y->ndata; i < l; i++)
			{
				ty = (float)rand() / (RAND_MAX + 1.0f);
				tx = (float)rand() / (RAND_MAX + 1.0f);
				py[i] = float16_to_float32(pdat->mean + pdat->scale * sqrtf(-2.0f * logf(tx)) * cos(2.0f * acos(-1.0f) * ty));
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			float ty, tx;

			for(i = 0, l = y->ndata; i < l; i++)
			{
				ty = (float)rand() / (RAND_MAX + 1.0f);
				tx = (float)rand() / (RAND_MAX + 1.0f);
				py[i] = pdat->mean + pdat->scale * sqrtf(-2.0f * logf(tx)) * cos(2.0f * acos(-1.0f) * ty);
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			double ty, tx;

			for(i = 0, l = y->ndata; i < l; i++)
			{
				ty = (double)rand() / (RAND_MAX + 1.0f);
				tx = (double)rand() / (RAND_MAX + 1.0f);
				py[i] = pdat->mean + pdat->scale * sqrt(-2.0f * log(tx)) * cos(2.0f * acos(-1.0f) * ty);
			}
		}
		break;
	default:
		break;
	}
}

void resolver_default_op_RandomNormalLike(struct onnx_node_t * n)
{
	if(n->opset >= 1)
	{
		n->init = RandomNormalLike_init;
		n->exit = RandomNormalLike_exit;
		n->reshape = RandomNormalLike_reshape;
		n->operator = RandomNormalLike_operator;
	}
}
