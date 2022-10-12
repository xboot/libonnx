#include "../onnx.h"

enum auto_pad_t {
	AUTO_PAD_NOTSET		= 0,
	AUTO_PAD_SAME_UPPER	= 1,
	AUTO_PAD_SAME_LOWER	= 2,
	AUTO_PAD_VALID		= 3,
};

struct operator_pdata_t {
	enum auto_pad_t auto_pad;
	int ceil_mode;
	int storage_order;
	int * kernels;
	int nkernel;
	int * dilations;
	int ndilation;
	int * pads;
	int npad;
	int * strides;
	int nstride;

	int cpads[32];
};

static int MaxPool_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	int64_t * ints;
	int i, l;

	if((n->ninput == 1) && (n->noutput >= 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			memset(pdat, 0, sizeof(struct operator_pdata_t));
			switch(shash(onnx_attribute_read_string(n, "auto_pad", "NOTSET")))
			{
			case 0xc3966fc2: /* "NOTSET" */
				pdat->auto_pad = AUTO_PAD_NOTSET;
				break;
			case 0xcbbc7856: /* "SAME_UPPER" */
				pdat->auto_pad = AUTO_PAD_SAME_UPPER;
				break;
			case 0xcb192d33: /* "SAME_LOWER" */
				pdat->auto_pad = AUTO_PAD_SAME_LOWER;
				break;
			case 0x0e382d15: /* "VALID" */
				pdat->auto_pad = AUTO_PAD_VALID;
				break;
			default:
				pdat->auto_pad = AUTO_PAD_NOTSET;
				break;
			}
			pdat->ceil_mode = onnx_attribute_read_int(n, "ceil_mode", 0);
			pdat->storage_order = onnx_attribute_read_int(n, "storage_order", 0);
			pdat->nkernel = onnx_attribute_read_ints(n, "kernel_shape", &ints);
			if(pdat->nkernel > 0)
			{
				pdat->kernels = malloc(sizeof(int) * pdat->nkernel);
				for(i = 0; i < pdat->nkernel; i++)
					pdat->kernels[i] = ints[i];
			}
			pdat->ndilation = pdat->nkernel;
			pdat->dilations = malloc(sizeof(int) * pdat->ndilation);
			if(pdat->dilations)
			{
				l = onnx_attribute_read_ints(n, "dilations", &ints);
				for(i = 0; i < l; i++)
					pdat->dilations[i] = ints[i];
				for(; i < pdat->ndilation; i++)
					pdat->dilations[i] = 1;
			}
			pdat->npad = pdat->nkernel * 2;
			pdat->pads = malloc(sizeof(int) * pdat->npad);
			if(pdat->pads)
			{
				l = onnx_attribute_read_ints(n, "pads", &ints);
				for(i = 0; i < l; i++)
					pdat->pads[i] = ints[i];
				for(; i < pdat->npad; i++)
					pdat->pads[i] = 0;
			}
			pdat->nstride = pdat->nkernel;
			pdat->strides = malloc(sizeof(int) * pdat->nstride);
			if(pdat->strides)
			{
				l = onnx_attribute_read_ints(n, "strides", &ints);
				for(i = 0; i < l; i++)
					pdat->strides[i] = ints[i];
				for(; i < pdat->nstride; i++)
					pdat->strides[i] = 1;
			}
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int MaxPool_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
	{
		if(pdat->kernels)
			free(pdat->kernels);
		if(pdat->dilations)
			free(pdat->dilations);
		if(pdat->pads)
			free(pdat->pads);
		if(pdat->strides)
			free(pdat->strides);
		free(pdat);
	}
	return 1;
}

static int MaxPool_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int ndim = x->ndim;
	int dims[ndim];
	int pad;
	int i;

	switch(pdat->auto_pad)
	{
	case AUTO_PAD_NOTSET:
		memcpy(pdat->cpads, pdat->pads, sizeof(int) * pdat->npad);
		break;
	case AUTO_PAD_SAME_UPPER:
		for(i = 0; i < pdat->npad / 2; i++)
		{
			pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) - x->dims[i + 2];
			pdat->cpads[i] = pad / 2;
			pdat->cpads[i + pdat->nkernel] = pad - pdat->cpads[i];
		}
		break;
	case AUTO_PAD_SAME_LOWER:
		for(i = 0; i < pdat->npad / 2; i++)
		{
			pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) - x->dims[i + 2];
			pdat->cpads[i + pdat->nkernel] = pad / 2;
			pdat->cpads[i] = pad - pdat->cpads[i + pdat->nkernel];
		}
		break;
	case AUTO_PAD_VALID:
		memset(pdat->cpads, 0, sizeof(int) * pdat->npad);
		break;
	default:
		break;
	}
	dims[0] = x->dims[0];
	dims[1] = x->dims[1];
	for(i = 0; i < ndim - 2; i++)
	{
		switch(pdat->auto_pad)
		{
		case AUTO_PAD_NOTSET:
			if(pdat->ceil_mode)
				dims[i + 2] = ceilf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->nkernel] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1)) / (float)pdat->strides[i] + 1);
			else
				dims[i + 2] = floorf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->nkernel] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1)) / (float)pdat->strides[i] + 1);
			break;
		case AUTO_PAD_SAME_UPPER:
		case AUTO_PAD_SAME_LOWER:
			dims[i + 2] = ceilf(x->dims[i + 2] / (float)pdat->strides[i]);
			break;
		case AUTO_PAD_VALID:
			dims[i + 2] = ceilf((x->dims[i + 2] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) + 1) / (float)pdat->strides[i]);
			break;
		default:
			break;
		}
	}
	return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static inline int dim_next(int ndim, int * dims, int * dim_max)
{
	if(ndim == 0)
		return 0;
	while(1)
	{
		ndim = ndim - 1;
		dims[ndim] += 1;
		if(dims[ndim] < dim_max[ndim])
			return 1;
		else
		{
			if(ndim == 0)
				return 0;
			dims[ndim] = 0;
		}
	}
}

static inline int dim_offset(int ndim, int * dims, int * dim_max)
{
	int o, s;
	int i;

	for(i = ndim - 1, o = 0, s = 1; i >= 0; i--)
	{
		o += dims[i] * s;
		s *= dim_max[i];
	}
	return o;
}

static void MaxPool_int8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int8_t * px = (int8_t *)x->datas;
	int8_t * py = (int8_t *)y->datas;
	int8_t maxv, v;
	int k_dim[x->ndim - 2];
	int i_dim[x->ndim];
	int o_dim[x->ndim];
	int b_dim[x->ndim];
	int i;

	memset(o_dim, 0, sizeof(o_dim));
	do {
		for(i = 2; i < x->ndim; ++i)
			b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
		maxv = INT8_MIN;
		memset(k_dim, 0, sizeof(k_dim));
		do {
			i_dim[0] = o_dim[0];
			i_dim[1] = o_dim[1];
			for(i = 2; i < x->ndim; ++i)
				i_dim[i] = b_dim[i] + k_dim[i - 2];
			for(i = 0; i < x->ndim; ++i)
			{
				if((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
					break;
			}
			if(i >= x->ndim)
			{
				v = px[dim_offset(x->ndim, i_dim, x->dims)];
				maxv = max(v, maxv);
			}
		} while(dim_next(x->ndim - 2, k_dim, pdat->kernels));
		py[dim_offset(x->ndim, o_dim, y->dims)] = maxv;
	} while(dim_next(x->ndim, o_dim, y->dims));
}

static void MaxPool_uint8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint8_t * px = (uint8_t *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t maxv, v;
	int k_dim[x->ndim - 2];
	int i_dim[x->ndim];
	int o_dim[x->ndim];
	int b_dim[x->ndim];
	int i;

	memset(o_dim, 0, sizeof(o_dim));
	do {
		for(i = 2; i < x->ndim; ++i)
			b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
		maxv = 0;
		memset(k_dim, 0, sizeof(k_dim));
		do {
			i_dim[0] = o_dim[0];
			i_dim[1] = o_dim[1];
			for(i = 2; i < x->ndim; ++i)
				i_dim[i] = b_dim[i] + k_dim[i - 2];
			for(i = 0; i < x->ndim; ++i)
			{
				if((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
					break;
			}
			if(i >= x->ndim)
			{
				v = px[dim_offset(x->ndim, i_dim, x->dims)];
				maxv = max(v, maxv);
			}
		} while(dim_next(x->ndim - 2, k_dim, pdat->kernels));
		py[dim_offset(x->ndim, o_dim, y->dims)] = maxv;
	} while(dim_next(x->ndim, o_dim, y->dims));
}

static void MaxPool_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float maxv, v;
	int k_dim[x->ndim - 2];
	int i_dim[x->ndim];
	int o_dim[x->ndim];
	int b_dim[x->ndim];
	int i;

	memset(o_dim, 0, sizeof(o_dim));
	do {
		for(i = 2; i < x->ndim; ++i)
			b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
		maxv = -FLT_MAX;
		memset(k_dim, 0, sizeof(k_dim));
		do {
			i_dim[0] = o_dim[0];
			i_dim[1] = o_dim[1];
			for(i = 2; i < x->ndim; ++i)
				i_dim[i] = b_dim[i] + k_dim[i - 2];
			for(i = 0; i < x->ndim; ++i)
			{
				if((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
					break;
			}
			if(i >= x->ndim)
			{
				v = float16_to_float32(px[dim_offset(x->ndim, i_dim, x->dims)]);
				maxv = fmaxf(v, maxv);
			}
		} while(dim_next(x->ndim - 2, k_dim, pdat->kernels));
		py[dim_offset(x->ndim, o_dim, y->dims)] = float32_to_float16(maxv);
	} while(dim_next(x->ndim, o_dim, y->dims));
}

static void MaxPool_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	float maxv, v;
	int k_dim[x->ndim - 2];
	int i_dim[x->ndim];
	int o_dim[x->ndim];
	int b_dim[x->ndim];
	int i;

	memset(o_dim, 0, sizeof(o_dim));
	do {
		for(i = 2; i < x->ndim; ++i)
			b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
		maxv = -FLT_MAX;
		memset(k_dim, 0, sizeof(k_dim));
		do {
			i_dim[0] = o_dim[0];
			i_dim[1] = o_dim[1];
			for(i = 2; i < x->ndim; ++i)
				i_dim[i] = b_dim[i] + k_dim[i - 2];
			for(i = 0; i < x->ndim; ++i)
			{
				if((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
					break;
			}
			if(i >= x->ndim)
			{
				v = px[dim_offset(x->ndim, i_dim, x->dims)];
				maxv = fmaxf(v, maxv);
			}
		} while(dim_next(x->ndim - 2, k_dim, pdat->kernels));
		py[dim_offset(x->ndim, o_dim, y->dims)] = maxv;
	} while(dim_next(x->ndim, o_dim, y->dims));
}

static void MaxPool_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	double maxv, v;
	int k_dim[x->ndim - 2];
	int i_dim[x->ndim];
	int o_dim[x->ndim];
	int b_dim[x->ndim];
	int i;

	memset(o_dim, 0, sizeof(o_dim));
	do {
		for(i = 2; i < x->ndim; ++i)
			b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
		maxv = -DBL_MAX;
		memset(k_dim, 0, sizeof(k_dim));
		do {
			i_dim[0] = o_dim[0];
			i_dim[1] = o_dim[1];
			for(i = 2; i < x->ndim; ++i)
				i_dim[i] = b_dim[i] + k_dim[i - 2];
			for(i = 0; i < x->ndim; ++i)
			{
				if((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
					break;
			}
			if(i >= x->ndim)
			{
				v = px[dim_offset(x->ndim, i_dim, x->dims)];
				maxv = fmax(v, maxv);
			}
		} while(dim_next(x->ndim - 2, k_dim, pdat->kernels));
		py[dim_offset(x->ndim, o_dim, y->dims)] = maxv;
	} while(dim_next(x->ndim, o_dim, y->dims));
}

void resolver_default_op_MaxPool(struct onnx_node_t * n)
{
	if(n->opset >= 12)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_int8;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_uint8;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float64;
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
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 10)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 8)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float64;
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
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->operator = MaxPool_float64;
			break;
		default:
			break;
		}
	}
}
