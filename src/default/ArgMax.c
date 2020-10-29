#include <onnx.h>

struct operator_pdata_t {
	int axis;
	int keepdims;
	int select_last_index;

	int dim;
	int stride;
};

static int ArgMax_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->axis = onnx_attribute_read_int(n, "axis", 0);
			pdat->keepdims = onnx_attribute_read_int(n, "keepdims", 1);
			pdat->select_last_index = onnx_attribute_read_int(n, "select_last_index", 0);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int ArgMax_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int ArgMax_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int axis = pdat->axis;
	int ndim = x->ndim;
	int dims[ndim];
	int i;

	if(axis < 0)
		axis += x->ndim;
	if(pdat->keepdims)
	{
		memcpy(dims, x->dims, sizeof(int) * ndim);
		dims[axis] = 1;
	}
	else
	{
		memcpy(dims, x->dims, sizeof(int) * ndim);
		dims[axis] = 1;
	}
	pdat->dim = x->dims[axis];
	pdat->stride = 1;
	for(i = ndim - 1; i > axis; i--)
		pdat->stride *= x->dims[i];
	return onnx_tensor_reshape(y, dims, ndim, ONNX_TENSOR_TYPE_INT64);
}

static void ArgMax_int8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int8_t * p, * px = x->datas;
	int8_t maxv;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_int16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int16_t * p, * px = x->datas;
	int16_t maxv;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_int32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int32_t * p, * px = x->datas;
	int32_t maxv;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_int64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int64_t * p, * px = x->datas;
	int64_t maxv;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_uint8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint8_t * p, * px = x->datas;
	uint8_t maxv;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_uint16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = x->datas;
	uint16_t maxv;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_uint32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint32_t * p, * px = x->datas;
	uint32_t maxv;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_uint64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint64_t * p, * px = x->datas;
	uint64_t maxv;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_bfloat16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = x->datas;
	float maxv, v;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = bfloat16_to_float32(px[idx]), maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				v = bfloat16_to_float32(*p);
				if(pdat->select_last_index)
				{
					if(v >= maxv)
					{
						maxv = v;
						maxi = i;
					}
				}
				else
				{
					if(v > maxv)
					{
						maxv = v;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = x->datas;
	float maxv, v;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = float16_to_float32(px[idx]), maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				v = float16_to_float32(*p);
				if(pdat->select_last_index)
				{
					if(v >= maxv)
					{
						maxv = v;
						maxi = i;
					}
				}
				else
				{
					if(v > maxv)
					{
						maxv = v;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * p, * px = x->datas;
	float maxv;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * p, * px = x->datas;
	double maxv;
	int64_t * py = y->datas;
	int64_t maxi;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

void resolver_default_op_ArgMax(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_INT8:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_int8;
		break;
	case ONNX_TENSOR_TYPE_INT16:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_int16;
		break;
	case ONNX_TENSOR_TYPE_INT32:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_int32;
		break;
	case ONNX_TENSOR_TYPE_INT64:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_int64;
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_uint8;
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_uint16;
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_uint32;
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_uint64;
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_bfloat16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
		n->operator = ArgMax_float64;
		break;
	default:
		break;
	}
}
