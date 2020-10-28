#include <onnx.h>

struct operator_pdata_t {
	int * perm;
	int nperm;
};

static int Transpose_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	struct onnx_tensor_t * x;
	struct onnx_tensor_t * y;
	int64_t * ints;
	int i, j;

	if((n->ninput > 0) && (n->noutput > 0))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->nperm = n->inputs[0]->ndim;
			pdat->perm = malloc(sizeof(int) * pdat->nperm);
			if(pdat->perm)
			{
				x = n->inputs[0];
				y = n->outputs[0];
				if(!onnx_tensor_shape_equal(y, x) || (y->type != x->type))
					onnx_tensor_reinit(y, x->type, x->dims, x->ndim);
				if(pdat->nperm == onnx_attribute_read_ints(n, "perm", &ints))
				{
					for(i = 0; i < pdat->nperm; i++)
						pdat->perm[i] = ints[i];
				}
				else
				{
					for(i = 0; i < pdat->nperm; i++)
						pdat->perm[i] = pdat->nperm - i - 1;
				}
				for(i = 0; i < n->noutput; i++)
				{
					for(j = 0; j < x->ndim; j++)
						n->outputs[i]->dims[j] = x->dims[pdat->perm[j]];
				}
				n->priv = pdat;
				return 1;
			}
			else
			{
				free(pdat);
			}
		}
	}
	return 0;
}

static int Transpose_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
	{
		if(pdat->perm)
			free(pdat->perm);
		free(pdat);
	}
	return 1;
}

static void Transpose_bool(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint8_t * px = (uint8_t *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_int8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int8_t * px = (int8_t *)x->datas;
	int8_t * py = (int8_t *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_int16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int16_t * px = (int16_t *)x->datas;
	int16_t * py = (int16_t *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_int32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int32_t * px = (int32_t *)x->datas;
	int32_t * py = (int32_t *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_int64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int64_t * px = (int64_t *)x->datas;
	int64_t * py = (int64_t *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_uint8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint8_t * px = (uint8_t *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_uint16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_uint32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint32_t * px = (uint32_t *)x->datas;
	uint32_t * py = (uint32_t *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_uint64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint64_t * px = (uint64_t *)x->datas;
	uint64_t * py = (uint64_t *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_bfloat16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
	}
}

static void Transpose_complex64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
		py[oy + 1] = px[ox + 1];
	}
}

static void Transpose_complex128(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		py[oy] = px[ox];
		py[oy + 1] = px[ox + 1];
	}
}

static void Transpose_string(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	char ** px = (char **)x->datas;
	char ** py = (char **)y->datas;
	int nperm = pdat->nperm;
	int ix[nperm], iy[nperm];
	int ox, oy;
	int i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, iy);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, ix);
		if(py[oy])
			free(py[oy]);
		py[oy] = strdup(px[ox]);
	}
}

void resolver_default_op_Transpose(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_bool;
		break;
	case ONNX_TENSOR_TYPE_INT8:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_int8;
		break;
	case ONNX_TENSOR_TYPE_INT16:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_int16;
		break;
	case ONNX_TENSOR_TYPE_INT32:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_int32;
		break;
	case ONNX_TENSOR_TYPE_INT64:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_int64;
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_uint8;
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_uint16;
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_uint32;
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_uint64;
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_bfloat16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_float64;
		break;
	case ONNX_TENSOR_TYPE_COMPLEX64:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_complex64;
		break;
	case ONNX_TENSOR_TYPE_COMPLEX128:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_complex128;
		break;
	case ONNX_TENSOR_TYPE_STRING:
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->operator = Transpose_string;
		break;
	default:
		break;
	}
}
