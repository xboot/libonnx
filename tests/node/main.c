#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <unistd.h>
#include <libgen.h>
#include <onnx.h>

#define FLOAT_PRECISION		(0.001f)

struct testcase_t {
	const char * name;
	const char * model_file;
	const char ** input_files;
	const char ** input_names;
	int ninput;
	const char ** output_files;
	const char ** output_names;
	int noutput;
};

static struct testcase_t testcases[] = {
	{
		.name			= "test_relu",
		.model_file		= "test_relu/model.onnx",
		.input_files	= &(const char *){ "test_relu/test_data_set_0/input_0.pb", },
		.input_names	= &(const char *){ "x", },
		.ninput			= 1,
		.output_files	= &(const char *){ "test_relu/test_data_set_0/output_0.pb", },
		.output_names	= &(const char *){ "y", },
		.noutput		= 1,
	},
	{
		.name			= "test_leakyrelu",
		.model_file		= "test_leakyrelu/model.onnx",
		.input_files	= &(const char *){ "test_leakyrelu/test_data_set_0/input_0.pb", },
		.input_names	= &(const char *){ "x", },
		.ninput			= 1,
		.output_files	= &(const char *){ "test_leakyrelu/test_data_set_0/output_0.pb", },
		.output_names	= &(const char *){ "y", },
		.noutput		= 1,
	},
	{
		.name			= "test_leakyrelu_default",
		.model_file		= "test_leakyrelu_default/model.onnx",
		.input_files	= &(const char *){ "test_leakyrelu_default/test_data_set_0/input_0.pb", },
		.input_names	= &(const char *){ "x", },
		.ninput			= 1,
		.output_files	= &(const char *){ "test_leakyrelu_default/test_data_set_0/output_0.pb", },
		.output_names	= &(const char *){ "y", },
		.noutput		= 1,
	},
	{
		.name			= "test_leakyrelu_example",
		.model_file		= "test_leakyrelu_example/model.onnx",
		.input_files	= &(const char *){ "test_leakyrelu_example/test_data_set_0/input_0.pb", },
		.input_names	= &(const char *){ "x", },
		.ninput			= 1,
		.output_files	= &(const char *){ "test_leakyrelu_example/test_data_set_0/output_0.pb", },
		.output_names	= &(const char *){ "y", },
		.noutput		= 1,
	},
};

static int onnx_tensor_equal(Onnx__TensorProto * a, Onnx__TensorProto * b)
{
	int i;

	if(!a || !b)
		return 0;
	if(a->data_type != b->data_type)
		return 0;
	if(a->n_dims != b->n_dims)
		return 0;
	if(memcmp(a->dims, b->dims, sizeof(int64_t) * a->n_dims) != 0)
		return 0;
	switch(a->data_type)
	{
	case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
		if(a->n_float_data != b->n_float_data)
			return 0;
		for(i = 0; i < a->n_float_data; i++)
		{
			if(fabs(a->float_data[i] - b->float_data[i]) > FLOAT_PRECISION)
				return 0;
		}
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
	return 1;
}

static void testcase_run(struct resolver_t * r, struct testcase_t * tc)
{
	struct onnx_context_t * ctx;
	Onnx__TensorProto * t, * o;
	int fail = 0;
	int i;

	ctx = onnx_context_alloc_from_file(tc->model_file, r);
	if(ctx)
	{
		for(i = 0; i < tc->ninput; i++)
		{
			t = onnx_search_tensor(ctx, tc->input_names[i]);
			o = onnx_tensor_alloc_from_file(tc->input_files[i]);
			onnx_tensor_copy(t, o);
			onnx_tensor_free(o);
		}
		onnx_run(ctx);
		for(i = 0; i < tc->noutput; i++)
		{
			t = onnx_search_tensor(ctx, tc->output_names[i]);
			o = onnx_tensor_alloc_from_file(tc->output_files[i]);
			if(!onnx_tensor_equal(t, o))
				fail |= 1;
			onnx_tensor_free(o);
		}
		onnx_context_free(ctx);
	}
	i = printf("\033[43;37m[%s]\033[0m", tc->name);
	printf("%*s\r\n", 80 + 12 - 6 - i, fail ? "\033[41;37m[FAIL]\033[0m" : "\033[42;37m[OKAY]\033[0m");
}

int main(int argc, char * argv[])
{
	struct resolver_t * r = NULL;
	char path[PATH_MAX];
	int i;

	if((readlink("/proc/self/exe", path, sizeof(path)) <= 0) || (chdir(dirname(path)) != 0))
		printf("ERROR: Can't change working directory.(%s)\r\n", getcwd(path, sizeof(path)));

	for(i = 0; i < (sizeof(testcases) / sizeof((testcases)[0])); i++)
		testcase_run(r, &testcases[i]);

	return 0;
}
