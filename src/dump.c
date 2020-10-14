#include <dump.h>

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

void onnx_dump_node(struct onnx_node_t * n)
{
	int i;

	printf("\tInput:\r\n");
	for(i = 0; i < n->ninput; i++)
	{
		printf("\t\t%s - ", n->inputs[i]->name);
		onnx_dump_tensor_type(n->inputs[i]);
		printf("\r\n");
	}
	printf("\tOutput:\r\n");
	for(i = 0; i < n->noutput; i++)
	{
		printf("\t\t%s - ", n->outputs[i]->name);
		onnx_dump_tensor_type(n->outputs[i]);
		printf("\r\n");
	}
	if(n->proto->n_attribute > 0)
	{
		printf("\tAttribute:\r\n");
		for(i = 0; i < n->proto->n_attribute; i++)
		{
			printf("\t\t%s - ", n->proto->attribute[i]->name);
			onnx_dump_attribute_type(n->proto->attribute[i]);
			printf("\r\n");
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
