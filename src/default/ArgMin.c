#include <onnx.h>

struct operator_pdata_t {
	int axis;
	int keepdims;
	int select_last_index;

	int dim;
	int stride;
};

static int ArgMin_init(struct onnx_node_t * n)
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

static int ArgMin_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int ArgMin_reshape(struct onnx_node_t * n)
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
	pdat->dim = x->dims[axis];
	pdat->stride = 1;
	for(i = ndim - 1; i > axis; i--)
		pdat->stride *= x->dims[i];
	if(pdat->keepdims)
	{
		memcpy(dims, x->dims, sizeof(int) * ndim);
		dims[axis] = 1;
	}
	else
	{
		for(i = 0, ndim = 0; i < x->ndim; i++)
		{
			if(i != axis)
				dims[ndim++]= x->dims[i];
		}
	}
	return onnx_tensor_reshape(y, dims, ndim, ONNX_TENSOR_TYPE_INT64);
}

static void ArgMin_int8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int8_t * p, * px = x->datas;
	int8_t minv;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_int16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int16_t * p, * px = x->datas;
	int16_t minv;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_int32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int32_t * p, * px = x->datas;
	int32_t minv;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_int64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int64_t * p, * px = x->datas;
	int64_t minv;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_uint8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint8_t * p, * px = x->datas;
	uint8_t minv;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_uint16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = x->datas;
	uint16_t minv;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_uint32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint32_t * p, * px = x->datas;
	uint32_t minv;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_uint64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint64_t * p, * px = x->datas;
	uint64_t minv;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_bfloat16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = x->datas;
	float minv, v;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = bfloat16_to_float32(px[idx]), mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				v = bfloat16_to_float32(*p);
				if(pdat->select_last_index)
				{
					if(v >= minv)
					{
						minv = v;
						mini = i;
					}
				}
				else
				{
					if(v > minv)
					{
						minv = v;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = x->datas;
	float minv, v;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = float16_to_float32(px[idx]), mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				v = float16_to_float32(*p);
				if(pdat->select_last_index)
				{
					if(v >= minv)
					{
						minv = v;
						mini = i;
					}
				}
				else
				{
					if(v > minv)
					{
						minv = v;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * p, * px = x->datas;
	float minv;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * p, * px = x->datas;
	double minv;
	int64_t * py = y->datas;
	int64_t mini;
	int len = x->ndata;
	int idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

void resolver_default_op_ArgMin(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_INT8:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_int8;
		break;
	case ONNX_TENSOR_TYPE_INT16:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_int16;
		break;
	case ONNX_TENSOR_TYPE_INT32:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_int32;
		break;
	case ONNX_TENSOR_TYPE_INT64:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_int64;
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_uint8;
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_uint16;
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_uint32;
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_uint64;
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_bfloat16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
		n->operator = ArgMin_float64;
		break;
	default:
		break;
	}
}
