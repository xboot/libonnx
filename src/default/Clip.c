#include <onnx.h>

union scalar_value_t {
	int8_t v_int8;
	int16_t v_int16;
	int32_t v_int32;
	int64_t v_int64;
	uint8_t v_uint8;
	uint16_t v_uint16;
	uint32_t v_uint32;
	uint64_t v_uint64;
	uint16_t v_bfloat16;
	uint16_t v_float16;
	float v_float32;
	double v_float64;
};

struct operator_pdata_t {
	union scalar_value_t u_min;
	union scalar_value_t u_max;
};

static void Clip_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	struct onnx_tensor_t * t = n->inputs[0];
	union scalar_value_t * v;
	int i;

	for(i = 0; i < n->noutput; i++)
	{
		if(n->outputs[i]->type == ONNX_TENSOR_TYPE_UNDEFINED)
			onnx_tensor_reinit(n->outputs[i], t->type, t->dims, t->ndim);
	}

	pdat = malloc(sizeof(struct operator_pdata_t));
	if(pdat)
	{
		switch(t->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			pdat->u_min.v_int8 = INT8_MIN;
			pdat->u_max.v_int8 = INT8_MAX;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			pdat->u_min.v_int16 = INT16_MIN;
			pdat->u_max.v_int16 = INT16_MAX;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			pdat->u_min.v_int32 = INT32_MIN;
			pdat->u_max.v_int32 = INT32_MAX;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			pdat->u_min.v_int64 = INT64_MIN;
			pdat->u_max.v_int64 = INT64_MAX;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			pdat->u_min.v_uint8 = 0;
			pdat->u_max.v_uint8 = UINT8_MAX;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			pdat->u_min.v_uint16 = 0;
			pdat->u_max.v_uint16 = UINT16_MAX;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			pdat->u_min.v_uint32 = 0;
			pdat->u_max.v_uint32 = UINT32_MAX;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			pdat->u_min.v_uint64 = 0;
			pdat->u_max.v_uint64 = UINT64_MAX;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			pdat->u_min.v_bfloat16 = float32_to_bfloat16(FLT_MIN);
			pdat->u_max.v_bfloat16 = float32_to_bfloat16(FLT_MAX);
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			pdat->u_min.v_float16 = float32_to_float16(FLT_MIN);
			pdat->u_max.v_float16 = float32_to_float16(FLT_MAX);
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			pdat->u_min.v_float32 = FLT_MIN;
			pdat->u_max.v_float32 = FLT_MAX;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			pdat->u_min.v_float64 = DBL_MIN;
			pdat->u_max.v_float64 = DBL_MAX;
			break;
		default:
			break;
		}
		for(i = 1; i < min(3, n->ninput); i++)
		{
			if(strcmp(n->inputs[i]->name, "min") == 0)
				v = &pdat->u_min;
			else if(strcmp(n->inputs[i]->name, "max") == 0)
				v = &pdat->u_max;
			else
				v = NULL;
			if(v && (n->inputs[i]->ndata > 0))
			{
				switch(t->type)
				{
				case ONNX_TENSOR_TYPE_INT8:
					v->v_int8 = ((int8_t *)n->inputs[i]->datas)[0];
					break;
				case ONNX_TENSOR_TYPE_INT16:
					v->v_int16 = ((int16_t *)n->inputs[i]->datas)[0];
					break;
				case ONNX_TENSOR_TYPE_INT32:
					v->v_int32 = ((int32_t *)n->inputs[i]->datas)[0];
					break;
				case ONNX_TENSOR_TYPE_INT64:
					v->v_int64 = ((int64_t *)n->inputs[i]->datas)[0];
					break;
				case ONNX_TENSOR_TYPE_UINT8:
					v->v_uint8 = ((uint8_t *)n->inputs[i]->datas)[0];
					break;
				case ONNX_TENSOR_TYPE_UINT16:
					v->v_uint16 = ((uint16_t *)n->inputs[i]->datas)[0];
					break;
				case ONNX_TENSOR_TYPE_UINT32:
					v->v_uint32 = ((uint32_t *)n->inputs[i]->datas)[0];
					break;
				case ONNX_TENSOR_TYPE_UINT64:
					v->v_uint64 = ((uint64_t *)n->inputs[i]->datas)[0];
					break;
				case ONNX_TENSOR_TYPE_BFLOAT16:
					v->v_bfloat16 = ((uint16_t *)n->inputs[i]->datas)[0];
					break;
				case ONNX_TENSOR_TYPE_FLOAT16:
					v->v_float16 = ((uint16_t *)n->inputs[i]->datas)[0];
					break;
				case ONNX_TENSOR_TYPE_FLOAT32:
					v->v_float32 = ((float *)n->inputs[i]->datas)[0];
					break;
				case ONNX_TENSOR_TYPE_FLOAT64:
					v->v_float64 = ((double *)n->inputs[i]->datas)[0];
					break;
				default:
					break;
				}
			}
		}
	}
	n->priv = pdat;
}

static void Clip_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
}

static void Clip_int8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int8_t * px = (int8_t *)x->datas;
	int8_t * py = (int8_t *)y->datas;
	int8_t minv = pdat->u_min.v_int8;
	int8_t maxv = pdat->u_max.v_int8;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] < minv)
			py[i] = minv;
		else if(px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

static void Clip_int16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int16_t * px = (int16_t *)x->datas;
	int16_t * py = (int16_t *)y->datas;
	int16_t minv = pdat->u_min.v_int16;
	int16_t maxv = pdat->u_max.v_int16;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] < minv)
			py[i] = minv;
		else if(px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

static void Clip_int32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int32_t * px = (int32_t *)x->datas;
	int32_t * py = (int32_t *)y->datas;
	int32_t minv = pdat->u_min.v_int32;
	int32_t maxv = pdat->u_max.v_int32;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] < minv)
			py[i] = minv;
		else if(px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

static void Clip_int64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int64_t * px = (int64_t *)x->datas;
	int64_t * py = (int64_t *)y->datas;
	int64_t minv = pdat->u_min.v_int64;
	int64_t maxv = pdat->u_max.v_int64;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] < minv)
			py[i] = minv;
		else if(px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

static void Clip_uint8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint8_t * px = (uint8_t *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t minv = pdat->u_min.v_uint8;
	uint8_t maxv = pdat->u_max.v_uint8;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] < minv)
			py[i] = minv;
		else if(px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

static void Clip_uint16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t minv = pdat->u_min.v_uint16;
	uint16_t maxv = pdat->u_max.v_uint16;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] < minv)
			py[i] = minv;
		else if(px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

static void Clip_uint32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint32_t * px = (uint32_t *)x->datas;
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t minv = pdat->u_min.v_uint32;
	uint32_t maxv = pdat->u_max.v_uint32;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] < minv)
			py[i] = minv;
		else if(px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

static void Clip_uint64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint64_t * px = (uint64_t *)x->datas;
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t minv = pdat->u_min.v_uint64;
	uint64_t maxv = pdat->u_max.v_uint64;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] < minv)
			py[i] = minv;
		else if(px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

static void Clip_bfloat16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float minv = bfloat16_to_float32(pdat->u_min.v_bfloat16);
	float maxv = bfloat16_to_float32(pdat->u_max.v_bfloat16);
	float v;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		if(v < minv)
			v = minv;
		else if(px[i] > maxv)
			v = maxv;
		else
			v = px[i];
		py[i] = float32_to_bfloat16(v);
	}
}

static void Clip_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float minv = float16_to_float32(pdat->u_min.v_float16);
	float maxv = float16_to_float32(pdat->u_max.v_float16);
	float v;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		if(v < minv)
			v = minv;
		else if(px[i] > maxv)
			v = maxv;
		else
			v = px[i];
		py[i] = float32_to_float16(v);
	}
}

static void Clip_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	float minv = pdat->u_min.v_float32;
	float maxv = pdat->u_max.v_float32;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] < minv)
			py[i] = minv;
		else if(px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

static void Clip_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	double minv = pdat->u_min.v_float64;
	double maxv = pdat->u_max.v_float64;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] < minv)
			py[i] = minv;
		else if(px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

void default_resolver_op_Clip(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_INT8:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_int8;
		break;
	case ONNX_TENSOR_TYPE_INT16:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_int16;
		break;
	case ONNX_TENSOR_TYPE_INT32:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_int32;
		break;
	case ONNX_TENSOR_TYPE_INT64:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_int64;
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_uint8;
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_uint16;
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_uint32;
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_uint64;
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_bfloat16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->op = Clip_float64;
		break;
	default:
		break;
	}
}
