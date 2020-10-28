#include <onnx.h>

struct operator_pdata_t {
	enum onnx_tensor_type_t type;
	union onnx_scalar_t scalar;
};

static int ConstantOfShape_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	Onnx__AttributeProto * attr;
	Onnx__TensorProto * t = NULL;
	int i;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			for(i = 0; i < n->proto->n_attribute; i++)
			{
				attr = n->proto->attribute[i];
				if((attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR) && (strcmp(attr->name, "value") == 0))
				{
					t = attr->t;
					break;
				}
			}
			if(t)
			{
				pdat->type = (enum onnx_tensor_type_t)t->data_type;
				switch(t->data_type)
				{
				case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
					pdat->scalar.v_float32 = t->float_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
					pdat->scalar.v_uint8 = t->int32_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
					pdat->scalar.v_int8 = t->int32_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
					pdat->scalar.v_uint16 = t->int32_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
					pdat->scalar.v_int16 = t->int32_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
					pdat->scalar.v_int32 = t->int32_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
					pdat->scalar.v_bool = t->int32_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
					pdat->scalar.v_float16 = t->int32_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
					pdat->scalar.v_bfloat16 = t->int32_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
					pdat->scalar.v_int64 = t->int64_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
					pdat->scalar.v_float64 = t->double_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
					pdat->scalar.v_uint32 = t->uint64_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
					pdat->scalar.v_uint64 = t->uint64_data[0];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
					pdat->scalar.v_complex64.real = t->float_data[0];
					pdat->scalar.v_complex64.imaginary = t->float_data[1];
					break;
				case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
					pdat->scalar.v_complex128.real = t->double_data[0];
					pdat->scalar.v_complex128.imaginary = t->double_data[1];
					break;
				default:
					memset(&pdat->scalar, 0, sizeof(union onnx_scalar_t));
					break;
				}
			}
			else
			{
				pdat->type = ONNX_TENSOR_TYPE_FLOAT32;
				memset(&pdat->scalar, 0, sizeof(union onnx_scalar_t));
			}
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int ConstantOfShape_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int ConstantOfShape_reshape(struct onnx_node_t * n)
{
	return 1;
}

static void ConstantOfShape_operator(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	char * p;
	int sz;
	int i, l;

	if(x->ndata > 0)
	{
		int dims[x->ndata];
		for(i = 0; i < x->ndata; i++)
			dims[i] = ((int64_t *)x->datas)[i];
		onnx_tensor_reinit(y, pdat->type, dims, x->ndata);
		for(i = 0, l = y->ndata, p = y->datas, sz = onnx_tensor_type_sizeof(pdat->type); i < l; i++, p += sz)
			memcpy(p, &pdat->scalar, sz);
	}
	else
	{
		onnx_tensor_reinit(y, pdat->type, NULL, 0);
		memcpy(&y->scalar, &pdat->scalar, sizeof(union onnx_scalar_t));
	}
}

void resolver_default_op_ConstantOfShape(struct onnx_node_t * n)
{
	n->init = ConstantOfShape_init;
	n->exit = ConstantOfShape_exit;
	n->reshape = ConstantOfShape_reshape;
	n->operator = ConstantOfShape_operator;
}
