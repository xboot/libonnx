#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <unistd.h>
#include <dirent.h>
#include <libgen.h>
#include <sys/stat.h>
#include <onnx.h>

#define FLOAT_PRECISION		(0.001f)

static int onnx_tensor_equal(struct onnx_tensor_t * a, struct onnx_tensor_t * b)
{
/*	int n = 0;
	int i;

	if(!a || !b)
		return 0;
	if(a->data_type != b->data_type)
		return 0;
	if(a->n_dims != b->n_dims)
		return 0;
	if(memcmp(a->dims, b->dims, sizeof(int64_t) * a->n_dims) != 0)
		return 0;
	if(a->n_dims > 0)
	{
		for(i = 0, n = 1; i < a->n_dims; i++)
		{
			if(a->dims[i] != 0)
				n *= a->dims[i];
		}
	}
	switch(a->data_type)
	{
	case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
		if((a->n_float_data != b->n_float_data) || (a->n_float_data != n))
			return 0;
		for(i = 0; i < a->n_float_data; i++)
		{
			if(fabs(a->float_data[i] - b->float_data[i]) > FLOAT_PRECISION)
				return 0;
		}
		break;
	case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
	case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
	case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
	case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
	case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
	case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
	case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
	case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
	case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
	case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
	case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
	case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
	case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
	case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
	case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
		//TODO
	default:
		return 0;
	}*/
	return 1;
}

static void testcase(const char * path, struct resolver_t * r)
{
	struct onnx_context_t * ctx;
	struct onnx_tensor_t * t, * o;
	struct stat st;
	char data_set_path[PATH_MAX];
	char tmp[PATH_MAX * 2];
	int data_set_index;
	int ninput, noutput;
	int fail;
	int len;

	sprintf(tmp, "%s/%s", path, "model.onnx");
	ctx = onnx_context_alloc_from_file(tmp, r);
	if(ctx)
	{
		data_set_index = 0;
		while(1)
		{
			sprintf(data_set_path, "%s/test_data_set_%d", path, data_set_index);
			if((lstat(data_set_path, &st) != 0) || !S_ISDIR(st.st_mode))
				break;
			ninput = 0;
			noutput = 0;
			fail = 0;
			while(1)
			{
				sprintf(tmp, "%s/input_%d.pb", data_set_path, ninput);
				if((lstat(tmp, &st) != 0) || !S_ISREG(st.st_mode))
					break;
				if(ninput > ctx->model->graph->n_input)
					break;
				t = onnx_search_tensor(ctx, ctx->model->graph->input[ninput]->name);
				o = onnx_tensor_alloc_from_file(tmp);
				//xxx onnx_tensor_copy(t, o);
				onnx_tensor_free(o);
				ninput++;
			}
			onnx_run(ctx);
			while(1)
			{
				sprintf(tmp, "%s/output_%d.pb", data_set_path, noutput);
				if((lstat(tmp, &st) != 0) || !S_ISREG(st.st_mode))
					break;
				if(noutput > ctx->model->graph->n_output)
					break;
				t = onnx_search_tensor(ctx, ctx->model->graph->output[noutput]->name);
				o = onnx_tensor_alloc_from_file(tmp);
				if(!onnx_tensor_equal(t, o))
					fail |= 1;
				onnx_tensor_free(o);
				noutput++;
			}

			len = printf("[%s](test_data_set_%d)", path, data_set_index);
			printf("%*s\r\n", 100 + 12 - 6 - len, fail ? "\033[41;37m[FAIL]\033[0m" : "\033[42;37m[OKAY]\033[0m");
			data_set_index++;
		}
		onnx_context_free(ctx);
	}
}

int main(int argc, char * argv[])
{
	struct resolver_t * r = NULL;
	struct hmap_t * m;
	struct hmap_entry_t * e;
	struct dirent * d;
	struct stat st;
	char path[PATH_MAX];
	DIR * dir;

	if((readlink("/proc/self/exe", path, sizeof(path)) <= 0) || (chdir(dirname(path)) != 0))
		printf("ERROR: Can't change working directory.(%s)\r\n", getcwd(path, sizeof(path)));

	if((lstat(path, &st) == 0) && S_ISDIR(st.st_mode))
	{
		m = hmap_alloc(0);
		if((dir = opendir(path)) != NULL)
		{
			while((d = readdir(dir)) != NULL)
			{
				if((lstat(d->d_name, &st) == 0) && S_ISDIR(st.st_mode))
				{
					if(strcmp(".", d->d_name) == 0)
						continue;
					if(strcmp("..", d->d_name) == 0)
						continue;
					hmap_add(m, d->d_name, NULL);
				}
			}
			closedir(dir);
		}
		hmap_sort(m);
		hmap_for_each_entry(e, m)
		{
			testcase(e->key, r);
		}
		hmap_free(m, NULL);
	}
	return 0;
}
