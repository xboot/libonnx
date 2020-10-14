#include <onnx.h>

static const char * data_type_tostring[] = {
	"undefined",
	"float32",
	"uint8",
	"int8",
	"uint16",
	"int16",
	"int32",
	"int64",
	"string",
	"bool",
	"float16",
	"float64",
	"uint32",
	"uint64",
	"complex64",
	"complex128",
	"bfloat16",
};

static const char * attribute_type_tostring[] = {
	"undefined",
	"float32",
	"int64",
	"string",
	"tensor",
	"graph",
	"float32[]",
	"int64[]",
	"string[]",
	"tensor[]",
	"graph[]",
	"sparse tensor",
	"sparse tensor[]",
};

static void onnx_dump_tensor_type(Onnx__TensorProto * t)
{
	int i;

	if(t)
	{
		printf("%s[", data_type_tostring[t->data_type]);
		for(i = 0; i < t->n_dims; i++)
		{
			printf("%ld", t->dims[i]);
			if(i != t->n_dims - 1)
				printf(" x ");
		}
		printf("]");
	}
}

static void onnx_dump_attribute_type(Onnx__AttributeProto * a)
{
	if(a)
		printf("%s", attribute_type_tostring[a->type]);
}

static void operator_stub(struct onnx_node_t * node)
{
	int i;

	printf("[%s] OPERATOR NOT IMPLEMENTED\r\n", node->np->op_type);
	printf("\tInput:\r\n");
	for(i = 0; i < node->ninput; i++)
	{
		printf("\t\t%s - ", node->input[i]->name);
		onnx_dump_tensor_type(node->input[i]);
		printf("\r\n");
	}
	printf("\tOutput:\r\n");
	for(i = 0; i < node->noutput; i++)
	{
		printf("\t\t%s - ", node->output[i]->name);
		onnx_dump_tensor_type(node->output[i]);
		printf("\r\n");
	}
	if(node->np->n_attribute > 0)
	{
		printf("\tAttribute:\r\n");
		for(i = 0; i < node->np->n_attribute; i++)
		{
			printf("\t\t%s - ", node->np->attribute[i]->name);
			onnx_dump_attribute_type(node->np->attribute[i]);
			printf("\r\n");
		}
	}
}

static Onnx__TensorProto * onnx_get_initializer(struct onnx_context_t * ctx, const char * name)
{
	Onnx__TensorProto ** initializer = ctx->model->graph->initializer;
	int i;

	for(i = 0; i < ctx->model->graph->n_initializer; i++)
	{
		if(strcmp(initializer[i]->name, name) == 0)
			return initializer[i];
	}
	return NULL;
}

static void hmap_entry_callback(struct hmap_entry_t * e)
{
	Onnx__TensorProto * t;

	if(e && (t = e->value))
		onnx_tensor_free(t);
}

struct onnx_context_t * onnx_context_alloc(const void * buf, int len)
{
	struct onnx_context_t * ctx;
	Onnx__TensorProto * t, * initial;
	Onnx__ValueInfoProto * v;
	char * name;
	int i, j;

	if(!buf || len <= 0)
		return NULL;

	ctx = malloc(sizeof(struct onnx_context_t));
	if(!ctx)
		return NULL;

	ctx->buflen = len;
	ctx->buf = memalign(8, ctx->buflen);
	if(!ctx->buf)
	{
		free(ctx);
		return NULL;
	}
	memcpy(ctx->buf, buf, ctx->buflen);

	ctx->model = onnx__model_proto__unpack(NULL, ctx->buflen, (const uint8_t *)ctx->buf);
	if(!ctx->model)
	{
		free(ctx->buf);
		free(ctx);
		return NULL;
	}

	ctx->map = hmap_alloc(0);
	if(!ctx->map)
	{
		onnx__model_proto__free_unpacked(ctx->model, NULL);
		free(ctx->buf);
		free(ctx);
		return NULL;
	}

	ctx->nlen = ctx->model->graph->n_node;
	ctx->node = malloc(sizeof(struct onnx_node_t) * ctx->nlen);
	if(!ctx->node)
	{
		hmap_free(ctx->map, hmap_entry_callback);
		onnx__model_proto__free_unpacked(ctx->model, NULL);
		free(ctx->buf);
		free(ctx);
		return NULL;
	}

	for(i = 0; i < ctx->model->graph->n_input; i++)
	{
		v = ctx->model->graph->input[i];
		if(!onnx_search_tensor(ctx, v->name))
		{
			t = onnx_tensor_alloc(v);
			if(t)
			{
				initial = onnx_get_initializer(ctx, t->name);
				if(initial)
				{
					//TODO Copy from initializer.
				}
				hmap_add(ctx->map, t->name, t);
			}
		}
	}

	for(i = 0; i < ctx->model->graph->n_output; i++)
	{
		v = ctx->model->graph->output[i];
		if(!onnx_search_tensor(ctx, v->name))
		{
			t = onnx_tensor_alloc(v);
			if(t)
				hmap_add(ctx->map, t->name, t);
		}
	}

	for(i = 0; i < ctx->model->graph->n_value_info; i++)
	{
		v = ctx->model->graph->value_info[i];
		if(!onnx_search_tensor(ctx, v->name))
		{
			t = onnx_tensor_alloc(v);
			if(t)
				hmap_add(ctx->map, t->name, t);
		}
	}

	for(i = 0; i < ctx->model->graph->n_node; i++)
	{
		for(j = 0; j < ctx->model->graph->node[i]->n_output; j++)
		{
			name = ctx->model->graph->node[i]->output[j];
			if(!onnx_search_tensor(ctx, name))
			{
				t = malloc(sizeof(Onnx__TensorProto));
				if(t)
				{
					onnx__tensor_proto__init(t);
					t->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED;
					t->name = strdup(name);
					t->doc_string = NULL;
					hmap_add(ctx->map, name, t);
				}
			}
		}
	}

	for(i = 0; i < ctx->model->graph->n_node; i++)
	{
		for(j = 0; j < ctx->model->graph->node[i]->n_input; j++)
		{
			name = ctx->model->graph->node[i]->input[j];
			if(!onnx_search_tensor(ctx, name))
			{
				free(ctx->node);
				hmap_free(ctx->map, hmap_entry_callback);
				onnx__model_proto__free_unpacked(ctx->model, NULL);
				free(ctx->buf);
				free(ctx);
				return NULL;
			}
		}
	}

	for(i = 0; i < ctx->nlen; i++)
	{
		ctx->node[i].ctx = ctx;
		ctx->node[i].np = ctx->model->graph->node[i];
		ctx->node[i].input = malloc(sizeof(Onnx__TensorProto *) * ctx->node[i].np->n_input);
		if(ctx->node[i].input)
		{
			ctx->node[i].ninput = ctx->node[i].np->n_input;
			for(j = 0; j < ctx->node[i].ninput; j++)
				ctx->node[i].input[j] = onnx_search_tensor(ctx, ctx->node[i].np->input[j]);
		}
		ctx->node[i].output = malloc(sizeof(Onnx__TensorProto *) * ctx->node[i].np->n_output);
		if(ctx->node[i].output)
		{
			ctx->node[i].noutput = ctx->node[i].np->n_output;
			for(j = 0; j < ctx->node[i].noutput; j++)
				ctx->node[i].output[j] = onnx_search_tensor(ctx, ctx->node[i].np->output[j]);
		}
		ctx->node[i].operator = NULL;
		if(!ctx->node[i].operator)
			ctx->node[i].operator = operator_stub;
	}

	return ctx;
}

struct onnx_context_t * onnx_context_alloc_from_file(const char * filename)
{
	struct onnx_context_t * ctx;
	FILE * fp;
    void * buf;
    int len;

	fp = fopen(filename, "rb");
	if(!fp)
		return NULL;

	fseek(fp, 0L, SEEK_END);
	len = ftell(fp);
	fseek(fp, 0L, SEEK_SET);

	buf = malloc(len);
	if(!buf)
	{
		fclose(fp);
		return NULL;
	}
	len = fread(buf, 1, len, fp);
	fclose(fp);

	ctx = onnx_context_alloc(buf, len);
	free(buf);
	return ctx;
}

void onnx_context_free(struct onnx_context_t * ctx)
{
	int i;

	if(ctx)
	{
		if(ctx->node)
		{
			for(i = 0; i < ctx->nlen; i++)
			{
				if(ctx->node[i].input)
					free(ctx->node[i].input);
				if(ctx->node[i].output)
					free(ctx->node[i].output);
			}
			free(ctx->node);
		}
		if(ctx->map)
			hmap_free(ctx->map, hmap_entry_callback);
		if(ctx->model)
			onnx__model_proto__free_unpacked(ctx->model, NULL);
		if(ctx->buf)
			free(ctx->buf);
		free(ctx);
	}
}

Onnx__TensorProto * onnx_tensor_alloc(Onnx__ValueInfoProto * v)
{
	Onnx__TensorProto * t;
	int n, i;

	if(!v)
		return NULL;

	t = malloc(sizeof(Onnx__TensorProto));
	if(!t)
		return NULL;

	onnx__tensor_proto__init(t);
	switch(v->type->value_case)
	{
	case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
		t->n_dims = v->type->tensor_type->shape->n_dim;
		t->dims = malloc(sizeof(int64_t) * t->n_dims);
		for(i = 0; i < t->n_dims; i++)
		{
			switch(v->type->tensor_type->shape->dim[i]->value_case)
			{
			case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
				t->dims[i] = v->type->tensor_type->shape->dim[i]->dim_value;
				break;
			case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
				break;
			default:
				break;
			}
		}
		t->data_type = v->type->tensor_type->elem_type;
		break;
	case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
		break;
	case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
		break;
	default:
		break;
	}
	if(t->n_dims > 0)
	{
		for(i = 0, n = 1; i < t->n_dims; i++)
			n *= t->dims[i];
		switch(t->data_type)
		{
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
			t->float_data = malloc(sizeof(float) * n);
			if(t->float_data)
			{
				memset(t->float_data, 0, sizeof(float) * n);
				t->n_float_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
			t->int32_data = malloc(sizeof(uint8_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(uint8_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
			t->int32_data = malloc(sizeof(int8_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int8_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
			t->int32_data = malloc(sizeof(uint16_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(uint16_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
			t->int32_data = malloc(sizeof(int16_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int16_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
			t->int32_data = malloc(sizeof(int32_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int32_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
			t->int64_data = malloc(sizeof(int64_t) * n);
			if(t->int64_data)
			{
				memset(t->int64_data, 0, sizeof(int64_t) * n);
				t->n_int64_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
			t->string_data = malloc(sizeof(ProtobufCBinaryData) * n);
			if(t->string_data)
			{
				for(i = 0; i < n; i++)
				{
					t->string_data[i].len = 0;
					t->string_data[i].data = NULL;
				}
				t->n_string_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
			t->int32_data = malloc(sizeof(int32_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int32_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
			t->int32_data = malloc(sizeof(int16_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int16_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
			t->double_data = malloc(sizeof(double) * n);
			if(t->double_data)
			{
				memset(t->double_data, 0, sizeof(double) * n);
				t->n_double_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
			t->uint64_data = malloc(sizeof(uint32_t) * n);
			if(t->int64_data)
			{
				memset(t->uint64_data, 0, sizeof(uint32_t) * n);
				t->n_uint64_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
			t->uint64_data = malloc(sizeof(uint64_t) * n);
			if(t->int64_data)
			{
				memset(t->uint64_data, 0, sizeof(uint64_t) * n);
				t->n_uint64_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
			t->float_data = malloc(sizeof(float) * 2 * n);
			if(t->float_data)
			{
				memset(t->float_data, 0, sizeof(float) *2 * n);
				t->n_float_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
			t->double_data = malloc(sizeof(double) * 2 * n);
			if(t->double_data)
			{
				memset(t->double_data, 0, sizeof(double) * 2 * n);
				t->n_double_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
			t->int32_data = malloc(sizeof(int16_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int16_t) * n);
				t->n_int32_data = n;
			}
			break;
		default:
			break;
		}
	}
	t->name = v->name ? strdup(v->name) : NULL;
	t->doc_string = v->doc_string ? strdup(v->doc_string) : NULL;
	return t;
}

void onnx_tensor_free(Onnx__TensorProto * t)
{
	int i;

	if(t)
	{
		if((t->n_dims > 0) && t->dims)
			free(t->dims);
		if(t->segment)
			free(t->segment);
		if((t->n_float_data > 0) && t->float_data)
			free(t->float_data);
		if((t->n_int32_data > 0) && t->int32_data)
			free(t->int32_data);
		if((t->n_string_data > 0) && t->string_data)
		{
			for(i = 0; i < t->n_string_data; i++)
			{
				if(t->string_data[i].data)
					free(t->string_data[i].data);
			}
			free(t->string_data);
		}
		if((t->n_int64_data > 0) && t->int64_data)
			free(t->int64_data);
		if(t->name)
			free(t->name);
		if(t->doc_string)
			free(t->doc_string);
		if((t->raw_data.len > 0) && t->raw_data.data)
			free(t->raw_data.data);
		if((t->n_external_data > 0) && t->external_data)
		{
			for(i = 0; i < t->n_external_data; i++)
			{
				if(t->external_data[i]->key)
					free(t->external_data[i]->key);
				if(t->external_data[i]->value)
					free(t->external_data[i]->value);
			}
			free(t->external_data);
		}
		if((t->n_double_data > 0) && t->double_data)
			free(t->double_data);
		if((t->n_uint64_data > 0) && t->uint64_data)
			free(t->uint64_data);
		free(t);
	}
}

Onnx__TensorProto * onnx_search_tensor(struct onnx_context_t * ctx, const char * name)
{
	if(ctx)
		return hmap_search(ctx->map, name);
	return NULL;
}

void onnx_solve(struct onnx_context_t * ctx)
{
	struct onnx_node_t * node;
	int i;

	if(ctx)
	{
		for(i = 0; i < ctx->nlen; i++)
		{
			node = &ctx->node[i];
			node->operator(node);
		}
	}
}

static Onnx__ValueInfoProto * onnx_get_input_value_info(struct onnx_context_t * ctx, const char * name)
{
	Onnx__ValueInfoProto ** input = ctx->model->graph->input;
	int i;

	for(i = 0; i < ctx->model->graph->n_input; i++)
	{
		if(strcmp(input[i]->name, name) == 0)
			return input[i];
	}
	return NULL;
}

static Onnx__ValueInfoProto * onnx_get_output_value_info(struct onnx_context_t * ctx, const char * name)
{
	Onnx__ValueInfoProto ** output = ctx->model->graph->output;
	int i;

	for(i = 0; i < ctx->model->graph->n_output; i++)
	{
		if(strcmp(output[i]->name, name) == 0)
			return output[i];
	}
	return NULL;
}

static Onnx__ValueInfoProto * onnx_get_value_info(struct onnx_context_t * ctx, const char * name)
{
	Onnx__ValueInfoProto ** value_info = ctx->model->graph->value_info;
	int i;

	for(i = 0; i < ctx->model->graph->n_value_info; i++)
	{
		if(strcmp(value_info[i]->name, name) == 0)
			return value_info[i];
	}
	return NULL;
}


static void onnx_dump_value_info_type(Onnx__ValueInfoProto * v)
{
	int i;

	if(v)
	{
		switch(v->type->value_case)
		{
		case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
			printf("%s[", data_type_tostring[v->type->tensor_type->elem_type]);
			for(i = 0; i < v->type->tensor_type->shape->n_dim; i++)
			{
				switch(v->type->tensor_type->shape->dim[i]->value_case)
				{
				case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
					printf("%ld", v->type->tensor_type->shape->dim[i]->dim_value);
					break;
				case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
					printf("%s", v->type->tensor_type->shape->dim[i]->dim_param);
					break;
				default:
					printf("?");
					break;
				}
				if(i != v->type->tensor_type->shape->n_dim - 1)
				{
					printf(" x ");
				}
			}
			printf("]");
			break;
		case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
			break;
		case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
			break;
		default:
			break;
		}
	}
}

void onnx_dump_model(struct onnx_context_t * ctx)
{
	Onnx__ModelProto * model;
	Onnx__TensorProto * t;
	Onnx__ValueInfoProto * v;
	int i, j;

	if(ctx && (model = ctx->model))
	{
		printf("IR Version: v%ld\r\n", model->ir_version);
		printf("Producer: %s %s\r\n", model->producer_name, model->producer_version);
		printf("Domain: %s\r\n", model->domain);
		printf("Imports:\r\n");
		for(i = 0; i < model->n_opset_import; i++)
			printf("\t%s v%ld\r\n", (strlen(model->opset_import[i]->domain) > 0) ? model->opset_import[i]->domain : "ai.onnx", model->opset_import[i]->version);

		printf("\r\nInitializer:\r\n");
		for(i = 0; i < model->graph->n_initializer; i++)
		{
			printf("\t%s - ", model->graph->initializer[i]->name);
			onnx_dump_tensor_type(model->graph->initializer[i]);
			printf("\r\n");
		}

		printf("\r\nInput:\r\n");
		for(i = 0; i < model->graph->n_input; i++)
		{
			printf("\t%s - ", model->graph->input[i]->name);
			onnx_dump_value_info_type(model->graph->input[i]);
			printf("\r\n");
		}

		printf("\r\nOutput:\r\n");
		for(i = 0; i < model->graph->n_output; i++)
		{
			printf("\t%s - ", model->graph->output[i]->name);
			onnx_dump_value_info_type(model->graph->output[i]);
			printf("\r\n");
		}

		printf("\r\nValueInfo:\r\n");
		for(i = 0; i < model->graph->n_value_info; i++)
		{
			printf("\t%s - ", model->graph->value_info[i]->name);
			onnx_dump_value_info_type(model->graph->value_info[i]);
			printf("\r\n");
		}

		printf("\r\nGraph node:\r\n");
		for(i = 0; i < model->graph->n_node; i++)
		{
			printf("[%s] - [%s]\r\n", model->graph->node[i]->name, model->graph->node[i]->op_type);
			printf("\tInputs:\r\n");
			for(j = 0; j < model->graph->node[i]->n_input; j++)
			{
				printf("\t\t%s - ", model->graph->node[i]->input[j]);
				v = onnx_get_input_value_info(ctx, model->graph->node[i]->input[j]);
				if(!v)
					v = onnx_get_value_info(ctx, model->graph->node[i]->input[j]);
				if(!v)
				{
					t = onnx_get_initializer(ctx, model->graph->node[i]->input[j]);
					if(t)
					{
						printf("initializer ");
						onnx_dump_tensor_type(t);
					}
				}
				else
					onnx_dump_value_info_type(v);
				printf("\r\n");
			}
			printf("\tOutputs:\r\n");
			for(j = 0; j < model->graph->node[i]->n_output; j++)
			{
				printf("\t\t%s - ", model->graph->node[i]->output[j]);
				v = onnx_get_output_value_info(ctx, model->graph->node[i]->output[j]);
				if(!v)
					v = onnx_get_value_info(ctx, model->graph->node[i]->output[j]);
				onnx_dump_value_info_type(v);
				printf("\r\n");
			}
			printf("\tAttributes:\r\n");
			for(j = 0; j < model->graph->node[i]->n_attribute; j++)
			{
				printf("\t\t%s - ", model->graph->node[i]->attribute[j]->name);
				onnx_dump_attribute_type(model->graph->node[i]->attribute[j]);
				printf("\r\n");
			}
		}
	}
}

void onnx_dump_tensor(Onnx__TensorProto * t)
{
	int n = 0, i;

	if(t)
	{
		printf("\r\n\r\n===============%s - ", t->name);
		onnx_dump_tensor_type(t);
		printf("\r\n");

		if(t->n_dims > 0)
		{
			for(i = 0, n = 1; i < t->n_dims; i++)
				n *= t->dims[i];
		}
		switch(t->data_type)
		{
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
			for(i = 0; i < n; i++)
				printf("%f\r\n", t->float_data[i]);
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
			break;
		default:
			break;
		}
	}
}
