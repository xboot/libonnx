#include <onnx.h>

static int Constant_init(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y;
	Onnx__AttributeProto * attr;

	if((n->noutput == 1) && (n->proto->n_attribute == 1))
	{
		y = n->outputs[0];
		attr = n->proto->attribute[0];
		if(attr)
		{
			switch(attr->type)
			{
			case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT:
				if(strcmp(attr->name, "value_float") == 0)
				{
					if((y->ndim != 0) || (y->type != ONNX_TENSOR_TYPE_FLOAT32))
						onnx_tensor_reinit(y, ONNX_TENSOR_TYPE_FLOAT32, NULL, 0);
					onnx_tensor_apply(y, &attr->f, sizeof(float));
					return 1;
				}
				break;
			case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT:
				if(strcmp(attr->name, "value_int") == 0)
				{
					if((y->ndim != 0) || (y->type != ONNX_TENSOR_TYPE_INT64))
						onnx_tensor_reinit(y, ONNX_TENSOR_TYPE_INT64, NULL, 0);
					onnx_tensor_apply(y, &attr->i, sizeof(int64_t));
					return 1;
				}
				break;
			case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING:
				break;
			case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS:
				if((strcmp(attr->name, "value_floats") == 0) && (attr->n_floats > 0))
				{
					if((y->ndim != 1) || (y->dims[0] != attr->n_floats) || (y->type != ONNX_TENSOR_TYPE_FLOAT32))
						onnx_tensor_reinit(y, ONNX_TENSOR_TYPE_FLOAT32, (int[]){ attr->n_floats }, 1);
					onnx_tensor_apply(y, attr->floats, attr->n_floats * sizeof(float));
					return 1;
				}
				break;
			case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS:
				if((strcmp(attr->name, "value_ints") == 0) && (attr->n_ints > 0))
				{
					if((y->ndim != 1) || (y->dims[0] != attr->n_ints) || (y->type != ONNX_TENSOR_TYPE_INT64))
						onnx_tensor_reinit(y, ONNX_TENSOR_TYPE_INT64, (int[]){ attr->n_ints }, 1);
					onnx_tensor_apply(y, attr->ints, attr->n_ints * sizeof(int64_t));
					return 1;
				}
				break;
			case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS:
				if((strcmp(attr->name, "value_strings") == 0) && (attr->n_strings > 0))
				{
					if((y->ndim != 1) || (y->dims[0] != attr->n_strings) || (y->type != ONNX_TENSOR_TYPE_STRING))
						onnx_tensor_reinit(y, ONNX_TENSOR_TYPE_STRING, (int[]){ attr->n_ints }, 1);
					if(y->datas && attr->strings)
					{
						char ** str = (char **)y->datas;
						for(size_t i = 0; i < y->ndata; i++)
						{
							if(str[i])
							{
								free(str[i]);
								str[i] = NULL;
							}
						}
						for(size_t i = 0; i < y->ndata; i++)
						{
							str[i] = malloc(attr->strings[i].len + 1);
							if(str[i])
							{
								str[i][attr->strings[i].len] = 0;
								memcpy(str[i], attr->strings[i].data, attr->strings[i].len);
							}
						}
					}
					return 1;
				}
				break;
			case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR:
				if(onnx_attribute_read_tensor(n, "value", y))
					return 1;
				break;
			case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR:
				break;
			default:
				break;
			}
		}
	}
	return 0;
}

static int Constant_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Constant_reshape(struct onnx_node_t * n)
{
	return 1;
}

static void Constant_operator(struct onnx_node_t * n)
{
}

void resolver_default_op_Constant(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		n->init = Constant_init;
		n->exit = Constant_exit;
		n->reshape = Constant_reshape;
		n->operator = Constant_operator;
	}
	else if(n->opset >= 12)
	{
		n->init = Constant_init;
		n->exit = Constant_exit;
		n->reshape = Constant_reshape;
		n->operator = Constant_operator;
	}
	else if(n->opset >= 11)
	{
		n->init = Constant_init;
		n->exit = Constant_exit;
		n->reshape = Constant_reshape;
		n->operator = Constant_operator;
	}
	else if(n->opset >= 9)
	{
		n->init = Constant_init;
		n->exit = Constant_exit;
		n->reshape = Constant_reshape;
		n->operator = Constant_operator;
	}
	else if(n->opset >= 1)
	{
		n->init = Constant_init;
		n->exit = Constant_exit;
		n->reshape = Constant_reshape;
		n->operator = Constant_operator;
	}
}
