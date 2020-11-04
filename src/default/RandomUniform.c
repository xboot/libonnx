#include <onnx.h>

struct operator_pdata_t {
	enum onnx_tensor_type_t dtype;
	float high;
	float low;
	float seed;
	int * shape;
	int nshape;
};

static int RandomUniform_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	int64_t * ints;
	int i;

	if(n->noutput == 1)
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->nshape = onnx_attribute_read_ints(n, "shape", &ints);
			if((pdat->nshape > 0) && (pdat->shape = malloc(sizeof(int) * pdat->nshape)))
			{
				pdat->dtype = (enum onnx_tensor_type_t)onnx_attribute_read_int(n, "dtype", 1);
				pdat->high = onnx_attribute_read_float(n, "high", 1.0);
				pdat->low = onnx_attribute_read_float(n, "low", 0.0);
				pdat->seed = onnx_attribute_read_float(n, "seed", 0.0);
				for(i = 0; i < pdat->nshape; i++)
					pdat->shape[i] = ints[i];
				n->priv = pdat;
				return 1;
			}
			else
			{
				free(pdat);
				return 0;
			}
		}
	}
	return 0;
}

static int RandomUniform_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
	{
		if(pdat->shape)
			free(pdat->shape);
		free(pdat);
	}
	return 1;
}

static int RandomUniform_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape(y, pdat->shape, pdat->nshape, pdat->dtype);
}

static void RandomUniform_operator(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	int i, l;

	if(pdat->seed != 0.0)
		srand(pdat->seed);
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

void resolver_default_op_RandomUniform(struct onnx_node_t * n)
{
	n->init = RandomUniform_init;
	n->exit = RandomUniform_exit;
	n->reshape = RandomUniform_reshape;
	n->operator = RandomUniform_operator;
}
