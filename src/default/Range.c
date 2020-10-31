#include <onnx.h>

struct operator_pdata_t {
	double start;
	double limit;
	double delta;
};

static int Range_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 3) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->start = 0;
			pdat->limit = 0;
			pdat->delta = 0;
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int Range_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static double scalar_get(union onnx_scalar_t * scalar, enum onnx_tensor_type_t type)
{
	double v;

	switch(type)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		v = scalar->v_bool;
		break;
	case ONNX_TENSOR_TYPE_INT8:
		v = scalar->v_int8;
		break;
	case ONNX_TENSOR_TYPE_INT16:
		v = scalar->v_int16;
		break;
	case ONNX_TENSOR_TYPE_INT32:
		v = scalar->v_int32;
		break;
	case ONNX_TENSOR_TYPE_INT64:
		v = scalar->v_int64;
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		v = scalar->v_uint8;
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		v = scalar->v_uint16;
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		v = scalar->v_uint32;
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		v = scalar->v_uint64;
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		v = bfloat16_to_float32(scalar->v_bfloat16);
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		v = float16_to_float32(scalar->v_bfloat16);
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		v = scalar->v_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		v = scalar->v_float64;
		break;
	default:
		v = 0;
		break;
	}
	return v;
}

static int Range_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	int ndim;

	pdat->start = scalar_get(&n->inputs[0]->scalar, n->inputs[0]->type);
	pdat->limit = scalar_get(&n->inputs[1]->scalar, n->inputs[1]->type);
	pdat->delta = scalar_get(&n->inputs[2]->scalar, n->inputs[2]->type);
	ndim = fmax(ceil((pdat->limit - pdat->start) / pdat->delta), 0);
	return onnx_tensor_reshape(y, (int[]){ ndim }, 1, n->inputs[0]->type);
}

static void Range_int16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	int16_t * py = (int16_t *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = pdat->start + (pdat->delta * i);
}

static void Range_int32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	int32_t * py = (int32_t *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = pdat->start + (pdat->delta * i);
}

static void Range_int64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	int64_t * py = (int64_t *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = pdat->start + (pdat->delta * i);
}

static void Range_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	float * py = (float *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = pdat->start + (pdat->delta * i);
}

static void Range_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	double * py = (double *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = pdat->start + (pdat->delta * i);
}

void resolver_default_op_Range(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_INT16:
		n->init = Range_init;
		n->exit = Range_exit;
		n->reshape = Range_reshape;
		n->operator = Range_int16;
		break;
	case ONNX_TENSOR_TYPE_INT32:
		n->init = Range_init;
		n->exit = Range_exit;
		n->reshape = Range_reshape;
		n->operator = Range_int32;
		break;
	case ONNX_TENSOR_TYPE_INT64:
		n->init = Range_init;
		n->exit = Range_exit;
		n->reshape = Range_reshape;
		n->operator = Range_int64;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = Range_init;
		n->exit = Range_exit;
		n->reshape = Range_reshape;
		n->operator = Range_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = Range_init;
		n->exit = Range_exit;
		n->reshape = Range_reshape;
		n->operator = Range_float64;
		break;
	default:
		break;
	}
}
