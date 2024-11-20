#include <onnx.h>

struct operator_pdata_t {
	enum onnx_tensor_type_t to;
};

static int Cast_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = onnx_malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->to = (enum onnx_tensor_type_t)onnx_attribute_read_int(n, "to", n->inputs[0]->type);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int Cast_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		onnx_free(pdat);
	return 1;
}

static int Cast_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, pdat->to);
}

static void Cast_bool(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint8_t * px = (uint8_t *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((px[i] != 0) ? 1.0 : 0.0);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((px[i] != 0) ? 1.0 : 0.0);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1.0 : 0.0;
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1.0 : 0.0;
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%u", (px[i] != 0) ? 1 : 0);
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_int8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int8_t * px = (int8_t *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%d", px[i]);
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_int16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int16_t * px = (int16_t *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%d", px[i]);
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_int32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int32_t * px = (int32_t *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%d", px[i]);
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_int64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int64_t * px = (int64_t *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%ld", px[i]);
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_uint8(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint8_t * px = (uint8_t *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%u", px[i]);
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_uint16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%u", px[i]);
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_uint32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint32_t * px = (uint32_t *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%u", px[i]);
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_uint64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint64_t * px = (uint64_t *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%lu", px[i]);
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_bfloat16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (bfloat16_to_float32(px[i]) != 0.0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)(bfloat16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)(bfloat16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)(bfloat16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)(bfloat16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)(bfloat16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)(bfloat16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)(bfloat16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)(bfloat16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((bfloat16_to_float32(px[i])));
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = bfloat16_to_float32(px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)(bfloat16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%g", bfloat16_to_float32(px[i]));
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (float16_to_float32(px[i]) != 0.0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)(float16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)(float16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)(float16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)(float16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)(float16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)(float16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)(float16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)(float16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((float16_to_float32(px[i])));
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float16_to_float32(px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)(float16_to_float32(px[i]));
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%g", float16_to_float32(px[i]));
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0.0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16(px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16(px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%g", px[i]);
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0.0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((float)px[i]);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			char buf[32];
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				sprintf(buf, "%g", px[i]);
				py[i] = onnx_strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

static void Cast_string(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	char ** px = (char **)x->datas;
	size_t i, l;

	switch(pdat->to)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)strtoul(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t * py = (int8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)strtol(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t * py = (int16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)strtol(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t * py = (int32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)strtol(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t * py = (int64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)strtoll(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t * py = (uint8_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)strtoul(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)strtoul(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t * py = (uint32_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)strtoul(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t * py = (uint64_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)strtoull(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_bfloat16((float)strtod(px[i], NULL));
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = float32_to_float16((float)strtod(px[i], NULL));
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)strtod(px[i], NULL);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)strtod(px[i], NULL);
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** py = (char **)y->datas;
			for(i = 0, l = y->ndata; i < l; i++)
			{
				if(py[i])
					onnx_free(py[i]);
				py[i] = onnx_strdup(px[i]);
			}
		}
		break;
	default:
		break;
	}
}

void resolver_default_op_Cast(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_bool;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_float64;
			break;
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_string;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 9)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_bool;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_float64;
			break;
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_string;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_bool;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Cast_init;
			n->exit = Cast_exit;
			n->reshape = Cast_reshape;
			n->operator = Cast_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
	}
}
