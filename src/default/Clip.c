#include <onnx.h>

struct operator_pdata_t {
	union onnx_scalar_t * pmin;
	union onnx_scalar_t * pmax;
};

static void Clip_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	struct onnx_tensor_t * t = n->inputs[0];
	int i;

	for(i = 0; i < n->noutput; i++)
	{
		if(n->outputs[i]->type == ONNX_TENSOR_TYPE_UNDEFINED)
			onnx_tensor_reinit(n->outputs[i], t->type, t->dims, t->ndim);
	}

	pdat = malloc(sizeof(struct operator_pdata_t));
	if(pdat)
	{
		pdat->pmin = NULL;
		pdat->pmax = NULL;
		for(i = 1; i < min(3, n->ninput); i++)
		{
			if(n->inputs[i]->ndata == 0)
			{
				if(strcmp(n->inputs[i]->name, "min") == 0)
					pdat->pmin = &n->inputs[i]->scalar;
				else if(strcmp(n->inputs[i]->name, "max") == 0)
					pdat->pmax = &n->inputs[i]->scalar;
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
	int8_t minv = pdat->pmin ? pdat->pmin->v_int8 : INT8_MIN;
	int8_t maxv = pdat->pmax ? pdat->pmax->v_int8 : INT8_MAX;
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
	int16_t minv = pdat->pmin ? pdat->pmin->v_int16 : INT16_MIN;
	int16_t maxv = pdat->pmax ? pdat->pmax->v_int16 : INT16_MAX;
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
	int32_t minv = pdat->pmin ? pdat->pmin->v_int32 : INT32_MIN;
	int32_t maxv = pdat->pmax ? pdat->pmax->v_int32 : INT32_MAX;
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
	int64_t minv = pdat->pmin ? pdat->pmin->v_int64 : INT64_MIN;
	int64_t maxv = pdat->pmax ? pdat->pmax->v_int64 : INT64_MAX;
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
	uint8_t minv = pdat->pmin ? pdat->pmin->v_uint8 : 0;
	uint8_t maxv = pdat->pmax ? pdat->pmax->v_uint8 : UINT8_MAX;
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
	uint16_t minv = pdat->pmin ? pdat->pmin->v_uint16 : 0;
	uint16_t maxv = pdat->pmax ? pdat->pmax->v_uint16 : UINT16_MAX;
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
	uint32_t minv = pdat->pmin ? pdat->pmin->v_uint32 : 0;
	uint32_t maxv = pdat->pmax ? pdat->pmax->v_uint32 : UINT32_MAX;
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
	uint64_t minv = pdat->pmin ? pdat->pmin->v_uint64 : 0;
	uint64_t maxv = pdat->pmax ? pdat->pmax->v_uint64 : UINT64_MAX;
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
	float minv = bfloat16_to_float32(pdat->pmin ? pdat->pmin->v_bfloat16 : float32_to_bfloat16(FLT_MIN));
	float maxv = bfloat16_to_float32(pdat->pmax ? pdat->pmax->v_bfloat16 : float32_to_bfloat16(FLT_MAX));
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
	float minv = float16_to_float32(pdat->pmin ? pdat->pmin->v_float16 : float32_to_float16(FLT_MIN));
	float maxv = float16_to_float32(pdat->pmax ? pdat->pmax->v_float16 : float32_to_float16(FLT_MAX));
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
	float minv = pdat->pmin ? pdat->pmin->v_float32 : FLT_MIN;
	float maxv = pdat->pmax ? pdat->pmax->v_float32 : FLT_MAX;
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
	double minv = pdat->pmin ? pdat->pmin->v_float64 : DBL_MIN;
	double maxv = pdat->pmax ? pdat->pmax->v_float64 : DBL_MAX;
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
