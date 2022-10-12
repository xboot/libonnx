#include "../onnx.h"

struct operator_pdata_t {
	enum onnx_tensor_type_t dtype;
	float high;
	float low;
	float seed;
};

static int RandomUniformLike_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->dtype = (enum onnx_tensor_type_t)onnx_attribute_read_int(n, "dtype", 0);
			pdat->high = onnx_attribute_read_float(n, "high", 1.0);
			pdat->low = onnx_attribute_read_float(n, "low", 0.0);
			pdat->seed = onnx_attribute_read_float(n, "seed", 0.0);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int RandomUniformLike_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int RandomUniformLike_reshape(struct onnx_node_t * n)
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

static void RandomUniformLike_operator(struct onnx_node_t * n)
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

			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float16_to_float32(((float)rand() / (float)RAND_MAX) * (pdat->high - pdat->low) + pdat->low);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;

			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = ((float)rand() / (float)RAND_MAX) * (pdat->high - pdat->low) + pdat->low;
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;

			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = ((double)rand() / (double)RAND_MAX) * (pdat->high - pdat->low) + pdat->low;
		}
		break;
	default:
		break;
	}
}

void resolver_default_op_RandomUniformLike(struct onnx_node_t * n)
{
	if(n->opset >= 1)
	{
		n->init = RandomUniformLike_init;
		n->exit = RandomUniformLike_exit;
		n->reshape = RandomUniformLike_reshape;
		n->operator = RandomUniformLike_operator;
	}
}
