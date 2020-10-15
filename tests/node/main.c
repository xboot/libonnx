#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <unistd.h>
#include <libgen.h>
#include <onnx.h>

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
};

static Onnx__TensorProto * pb_tensor_alloc(const char * filename)
{
	Onnx__TensorProto * t = NULL;
	FILE * fp;
	void * buf;
	size_t l, len;

	fp = fopen(filename, "rb");
	if(fp)
	{
		fseek(fp, 0L, SEEK_END);
		l = ftell(fp);
		fseek(fp, 0L, SEEK_SET);
		if(l > 0)
		{
			buf = malloc(l);
			if(buf)
			{
				for(len = 0; len < l; len += fread(buf + len, 1, l - len, fp));
				t = onnx__tensor_proto__unpack(NULL, len, buf);
				free(buf);
			}
		}
	}
	fclose(fp);
	return t;
}

static void pb_tensor_free(Onnx__TensorProto * t)
{
	if(t)
		onnx__tensor_proto__free_unpacked(t, NULL);
}

static int testcase_run(struct resolver_t * r, struct testcase_t * tc)
{
	struct onnx_context_t * ctx;
	Onnx__TensorProto * t;

	if(!tc)
		return 0;

	ctx = onnx_context_alloc_from_file(tc->model_file, r);
	if(ctx)
	{
		t = pb_tensor_alloc(tc->input_files[0]);
		pb_tensor_free(t);

		onnx_dump_model(ctx);
	}
	return 1;
}

static void testcase_print(const char * name, int result)
{
	int len = printf("\033[43;37m[%s]\033[0m", name);
	printf("%*s\r\n", 80 + 12 - 6 - len, result ? "\033[42;37m[OKAY]\033[0m" : "\033[41;37m[FAIL]\033[0m");
}

int main(int argc, char * argv[])
{
	struct resolver_t * r = NULL;
	struct testcase_t * tc;
	char path[PATH_MAX];
	int i;

	if((readlink("/proc/self/exe", path, sizeof(path)) <= 0) || (chdir(dirname(path)) != 0))
		printf("ERROR: Can't change working directory.(%s)\r\n", getcwd(path, sizeof(path)));

	for(i = 0; i < (sizeof(testcases) / sizeof((testcases)[0])); i++)
	{
		tc = &testcases[i];
		testcase_print(tc->name, testcase_run(r, tc));
	}
	return 0;
}
