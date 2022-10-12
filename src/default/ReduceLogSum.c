#include "../onnx.h"

struct operator_pdata_t {
	int * axes;
	int naxes;
	int keepdims;

	int * caxes;
};

static int ReduceLogSum_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	int64_t * ints;
	int nint;
	int i;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			nint = onnx_attribute_read_ints(n, "axes", &ints);
			if(nint > 0)
				pdat->naxes = nint;
			else
				pdat->naxes = n->inputs[0]->ndim;
			pdat->axes = malloc(sizeof(int) * pdat->naxes);
			pdat->caxes = malloc(sizeof(int) * pdat->naxes);
			if(pdat->axes && pdat->caxes)
			{
				if(nint > 0)
				{
					for(i = 0; i < pdat->naxes; i++)
						pdat->axes[i] = ints[i];
				}
				else
				{
					for(i = 0; i < pdat->naxes; i++)
						pdat->axes[i] = i;
				}
				pdat->keepdims = onnx_attribute_read_int(n, "keepdims", 1);
				n->priv = pdat;
				return 1;
			}
			else
			{
				if(pdat->axes)
					free(pdat->axes);
				if(pdat->caxes)
					free(pdat->caxes);
				free(pdat);
			}
		}
	}
	return 0;
}

static int ReduceLogSum_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
	{
		if(pdat->axes)
			free(pdat->axes);
		if(pdat->caxes)
			free(pdat->caxes);
		free(pdat);
	}
	return 1;
}

static int ReduceLogSum_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int ndim = x->ndim;
	int dims[ndim];
	int axis, found;
	int i, j;

	for(i = 0; i < pdat->naxes; i++)
	{
		axis = pdat->axes[i];
		if(axis < 0)
			axis += x->ndim;
		if(axis < 0 || axis >= x->ndim)
			return 0;
		pdat->caxes[i] = axis;
	}
	if(pdat->keepdims)
	{
		memcpy(dims, x->dims, sizeof(int) * ndim);
		for(i = 0; i < pdat->naxes; i++)
			dims[pdat->caxes[i]] = 1;
	}
	else
	{
		for(i = 0, ndim = 0; i < x->ndim; i++)
		{
			for(j = 0, found = 0; j < pdat->naxes; j++)
			{
				if(i == pdat->caxes[j])
				{
					found = 1;
					break;
				}
			}
			if(!found)
				dims[ndim++]= x->dims[i];
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

static inline int dim_offset(int ndim, int * dims, int * distance)
{
	int i, o;

	if(ndim == 0)
		return 0;
	for(i = ndim - 1, o = 0; i >= 0; i--)
		o += dims[i] * distance[i];
	return o;
}

static void ReduceLogSum_int8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int8_t * px = (int8_t *)x->datas;
	int8_t * py = (int8_t *)y->datas;
	float sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	int iter_not_in_axes_max[not_in_axes_num];
	int iter_not_in_axes[not_in_axes_num];
	int not_in_axes_axis_dis[x->ndim];
	int iter_in_axes_max[pdat->naxes];
	int in_axes_axis_dis[pdat->naxes];
	int iter_in_axes[pdat->naxes];
	uint32_t mask;
	int i, j, k, o;

	for(i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for(i = 0, j = 0, k = 0; i < x->ndim; i++)
	{
		if(mask & (1 << i))
		{
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(iter_not_in_axes, 0, sizeof(int) * not_in_axes_num);
	do
	{
		memset(iter_in_axes, 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, iter_not_in_axes, not_in_axes_axis_dis);
		sum = 0;
		do
		{
			sum += px[o + dim_offset(pdat->naxes, iter_in_axes, in_axes_axis_dis)];
		} while(dim_next(pdat->naxes, iter_in_axes, iter_in_axes_max));
		py[i++] = logf(sum);
	} while(dim_next(not_in_axes_num, iter_not_in_axes, iter_not_in_axes_max));
}

static void ReduceLogSum_int32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int32_t * px = (int32_t *)x->datas;
	int32_t * py = (int32_t *)y->datas;
	float sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	int iter_not_in_axes_max[not_in_axes_num];
	int iter_not_in_axes[not_in_axes_num];
	int not_in_axes_axis_dis[x->ndim];
	int iter_in_axes_max[pdat->naxes];
	int in_axes_axis_dis[pdat->naxes];
	int iter_in_axes[pdat->naxes];
	uint32_t mask;
	int i, j, k, o;

	for(i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for(i = 0, j = 0, k = 0; i < x->ndim; i++)
	{
		if(mask & (1 << i))
		{
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(iter_not_in_axes, 0, sizeof(int) * not_in_axes_num);
	do
	{
		memset(iter_in_axes, 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, iter_not_in_axes, not_in_axes_axis_dis);
		sum = 0;
		do
		{
			sum += px[o + dim_offset(pdat->naxes, iter_in_axes, in_axes_axis_dis)];
		} while(dim_next(pdat->naxes, iter_in_axes, iter_in_axes_max));
		py[i++] = logf(sum);
	} while(dim_next(not_in_axes_num, iter_not_in_axes, iter_not_in_axes_max));
}

static void ReduceLogSum_int64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int64_t * px = (int64_t *)x->datas;
	int64_t * py = (int64_t *)y->datas;
	float sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	int iter_not_in_axes_max[not_in_axes_num];
	int iter_not_in_axes[not_in_axes_num];
	int not_in_axes_axis_dis[x->ndim];
	int iter_in_axes_max[pdat->naxes];
	int in_axes_axis_dis[pdat->naxes];
	int iter_in_axes[pdat->naxes];
	uint32_t mask;
	int i, j, k, o;

	for(i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for(i = 0, j = 0, k = 0; i < x->ndim; i++)
	{
		if(mask & (1 << i))
		{
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(iter_not_in_axes, 0, sizeof(int) * not_in_axes_num);
	do
	{
		memset(iter_in_axes, 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, iter_not_in_axes, not_in_axes_axis_dis);
		sum = 0;
		do
		{
			sum += px[o + dim_offset(pdat->naxes, iter_in_axes, in_axes_axis_dis)];
		} while(dim_next(pdat->naxes, iter_in_axes, iter_in_axes_max));
		py[i++] = logf(sum);
	} while(dim_next(not_in_axes_num, iter_not_in_axes, iter_not_in_axes_max));
}

static void ReduceLogSum_uint8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint8_t * px = (uint8_t *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;
	float sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	int iter_not_in_axes_max[not_in_axes_num];
	int iter_not_in_axes[not_in_axes_num];
	int not_in_axes_axis_dis[x->ndim];
	int iter_in_axes_max[pdat->naxes];
	int in_axes_axis_dis[pdat->naxes];
	int iter_in_axes[pdat->naxes];
	uint32_t mask;
	int i, j, k, o;

	for(i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for(i = 0, j = 0, k = 0; i < x->ndim; i++)
	{
		if(mask & (1 << i))
		{
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(iter_not_in_axes, 0, sizeof(int) * not_in_axes_num);
	do
	{
		memset(iter_in_axes, 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, iter_not_in_axes, not_in_axes_axis_dis);
		sum = 0;
		do
		{
			sum += px[o + dim_offset(pdat->naxes, iter_in_axes, in_axes_axis_dis)];
		} while(dim_next(pdat->naxes, iter_in_axes, iter_in_axes_max));
		py[i++] = logf(sum);
	} while(dim_next(not_in_axes_num, iter_not_in_axes, iter_not_in_axes_max));
}

static void ReduceLogSum_uint32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint32_t * px = (uint32_t *)x->datas;
	uint32_t * py = (uint32_t *)y->datas;
	float sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	int iter_not_in_axes_max[not_in_axes_num];
	int iter_not_in_axes[not_in_axes_num];
	int not_in_axes_axis_dis[x->ndim];
	int iter_in_axes_max[pdat->naxes];
	int in_axes_axis_dis[pdat->naxes];
	int iter_in_axes[pdat->naxes];
	uint32_t mask;
	int i, j, k, o;

	for(i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for(i = 0, j = 0, k = 0; i < x->ndim; i++)
	{
		if(mask & (1 << i))
		{
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(iter_not_in_axes, 0, sizeof(int) * not_in_axes_num);
	do
	{
		memset(iter_in_axes, 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, iter_not_in_axes, not_in_axes_axis_dis);
		sum = 0;
		do
		{
			sum += px[o + dim_offset(pdat->naxes, iter_in_axes, in_axes_axis_dis)];
		} while(dim_next(pdat->naxes, iter_in_axes, iter_in_axes_max));
		py[i++] = logf(sum);
	} while(dim_next(not_in_axes_num, iter_not_in_axes, iter_not_in_axes_max));
}

static void ReduceLogSum_uint64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint64_t * px = (uint64_t *)x->datas;
	uint64_t * py = (uint64_t *)y->datas;
	float sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	int iter_not_in_axes_max[not_in_axes_num];
	int iter_not_in_axes[not_in_axes_num];
	int not_in_axes_axis_dis[x->ndim];
	int iter_in_axes_max[pdat->naxes];
	int in_axes_axis_dis[pdat->naxes];
	int iter_in_axes[pdat->naxes];
	uint32_t mask;
	int i, j, k, o;

	for(i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for(i = 0, j = 0, k = 0; i < x->ndim; i++)
	{
		if(mask & (1 << i))
		{
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(iter_not_in_axes, 0, sizeof(int) * not_in_axes_num);
	do
	{
		memset(iter_in_axes, 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, iter_not_in_axes, not_in_axes_axis_dis);
		sum = 0;
		do
		{
			sum += px[o + dim_offset(pdat->naxes, iter_in_axes, in_axes_axis_dis)];
		} while(dim_next(pdat->naxes, iter_in_axes, iter_in_axes_max));
		py[i++] = logf(sum);
	} while(dim_next(not_in_axes_num, iter_not_in_axes, iter_not_in_axes_max));
}

static void ReduceLogSum_bfloat16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	int iter_not_in_axes_max[not_in_axes_num];
	int iter_not_in_axes[not_in_axes_num];
	int not_in_axes_axis_dis[x->ndim];
	int iter_in_axes_max[pdat->naxes];
	int in_axes_axis_dis[pdat->naxes];
	int iter_in_axes[pdat->naxes];
	uint32_t mask;
	int i, j, k, o;

	for(i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for(i = 0, j = 0, k = 0; i < x->ndim; i++)
	{
		if(mask & (1 << i))
		{
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(iter_not_in_axes, 0, sizeof(int) * not_in_axes_num);
	do
	{
		memset(iter_in_axes, 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, iter_not_in_axes, not_in_axes_axis_dis);
		sum = 0;
		do
		{
			sum += bfloat16_to_float32(px[o + dim_offset(pdat->naxes, iter_in_axes, in_axes_axis_dis)]);
		} while(dim_next(pdat->naxes, iter_in_axes, iter_in_axes_max));
		py[i++] = float32_to_bfloat16(logf(sum));
	} while(dim_next(not_in_axes_num, iter_not_in_axes, iter_not_in_axes_max));
}

static void ReduceLogSum_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	int iter_not_in_axes_max[not_in_axes_num];
	int iter_not_in_axes[not_in_axes_num];
	int not_in_axes_axis_dis[x->ndim];
	int iter_in_axes_max[pdat->naxes];
	int in_axes_axis_dis[pdat->naxes];
	int iter_in_axes[pdat->naxes];
	uint32_t mask;
	int i, j, k, o;

	for(i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for(i = 0, j = 0, k = 0; i < x->ndim; i++)
	{
		if(mask & (1 << i))
		{
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(iter_not_in_axes, 0, sizeof(int) * not_in_axes_num);
	do
	{
		memset(iter_in_axes, 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, iter_not_in_axes, not_in_axes_axis_dis);
		sum = 0;
		do
		{
			sum += float16_to_float32(px[o + dim_offset(pdat->naxes, iter_in_axes, in_axes_axis_dis)]);
		} while(dim_next(pdat->naxes, iter_in_axes, iter_in_axes_max));
		py[i++] = float32_to_float16(logf(sum));
	} while(dim_next(not_in_axes_num, iter_not_in_axes, iter_not_in_axes_max));
}

static void ReduceLogSum_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	float sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	int iter_not_in_axes_max[not_in_axes_num];
	int iter_not_in_axes[not_in_axes_num];
	int not_in_axes_axis_dis[x->ndim];
	int iter_in_axes_max[pdat->naxes];
	int in_axes_axis_dis[pdat->naxes];
	int iter_in_axes[pdat->naxes];
	uint32_t mask;
	int i, j, k, o;

	for(i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for(i = 0, j = 0, k = 0; i < x->ndim; i++)
	{
		if(mask & (1 << i))
		{
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(iter_not_in_axes, 0, sizeof(int) * not_in_axes_num);
	do
	{
		memset(iter_in_axes, 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, iter_not_in_axes, not_in_axes_axis_dis);
		sum = 0;
		do
		{
			sum += px[o + dim_offset(pdat->naxes, iter_in_axes, in_axes_axis_dis)];
		} while(dim_next(pdat->naxes, iter_in_axes, iter_in_axes_max));
		py[i++] = logf(sum);
	} while(dim_next(not_in_axes_num, iter_not_in_axes, iter_not_in_axes_max));
}

static void ReduceLogSum_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	double sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	int iter_not_in_axes_max[not_in_axes_num];
	int iter_not_in_axes[not_in_axes_num];
	int not_in_axes_axis_dis[x->ndim];
	int iter_in_axes_max[pdat->naxes];
	int in_axes_axis_dis[pdat->naxes];
	int iter_in_axes[pdat->naxes];
	uint32_t mask;
	int i, j, k, o;

	for(i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for(i = 0, j = 0, k = 0; i < x->ndim; i++)
	{
		if(mask & (1 << i))
		{
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(iter_not_in_axes, 0, sizeof(int) * not_in_axes_num);
	do
	{
		memset(iter_in_axes, 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, iter_not_in_axes, not_in_axes_axis_dis);
		sum = 0;
		do
		{
			sum += px[o + dim_offset(pdat->naxes, iter_in_axes, in_axes_axis_dis)];
		} while(dim_next(pdat->naxes, iter_in_axes, iter_in_axes_max));
		py[i++] = log(sum);
	} while(dim_next(not_in_axes_num, iter_not_in_axes, iter_not_in_axes_max));
}

void resolver_default_op_ReduceLogSum(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_int8;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 11)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_int8;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_int8;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ReduceLogSum_init;
			n->exit = ReduceLogSum_exit;
			n->reshape = ReduceLogSum_reshape;
			n->operator = ReduceLogSum_float64;
			break;
		default:
			break;
		}
	}
}
