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

static void testcase(const char * path, struct onnx_resolver_t ** r, int rlen)
{
	struct onnx_context_t * ctx;
	struct onnx_tensor_t * t, * o;
	struct stat st;
	char data_set_path[PATH_MAX];
	char tmp[PATH_MAX * 2];
	int data_set_index;
	int ninput, noutput;
	int okay;
	int len;

	sprintf(tmp, "%s/%s", path, "model.onnx");
	ctx = onnx_context_alloc_from_file(tmp, r, rlen);
	if(ctx)
	{
		data_set_index = 0;
		while(1)
		{
			sprintf(data_set_path, "%s/test_data_set_%d", path, data_set_index);
			if((stat(data_set_path, &st) != 0) || !S_ISDIR(st.st_mode))
				break;
			ninput = 0;
			noutput = 0;
			okay = 0;
			while(1)
			{
				sprintf(tmp, "%s/input_%d.pb", data_set_path, ninput);
				if((stat(tmp, &st) != 0) || !S_ISREG(st.st_mode))
					break;
				if(ninput > ctx->model->graph->n_input)
					break;
				t = onnx_tensor_search(ctx, ctx->model->graph->input[ninput]->name);
				o = onnx_tensor_alloc_from_file(tmp);
				onnx_tensor_apply(t, o->datas, o->ndata * onnx_tensor_type_sizeof(o->type));
				onnx_tensor_free(o);
				okay++;
				ninput++;
			}
			onnx_run(ctx);
			while(1)
			{
				sprintf(tmp, "%s/output_%d.pb", data_set_path, noutput);
				if((stat(tmp, &st) != 0) || !S_ISREG(st.st_mode))
					break;
				if(noutput > ctx->model->graph->n_output)
					break;
				t = onnx_tensor_search(ctx, ctx->model->graph->output[noutput]->name);
				o = onnx_tensor_alloc_from_file(tmp);
				if(onnx_tensor_equal(t, o))
					okay++;
				onnx_tensor_free(o);
				noutput++;
			}

			len = printf("[%s](test_data_set_%d)", path, data_set_index);
			printf("%*s\r\n", 100 + 12 - 6 - len, ((ninput + noutput == okay) && (okay > 0)) ? "\033[42;37m[OKAY]\033[0m" : "\033[41;37m[FAIL]\033[0m");
			data_set_index++;
		}
		onnx_context_free(ctx);
	}
	else
	{
		len = printf("[%s]", path);
		printf("%*s\r\n", 100 + 12 - 6 - len, "\033[41;37m[FAIL]\033[0m");
	}
}

static void usage(void)
{
	printf("usage:\r\n");
	printf("    tests <DIRECTORY>\r\n");
	printf("examples:\r\n");
	printf("    tests ./tests/model\r\n");
	printf("    tests ./tests/node\r\n");
	printf("    tests ./tests/pytorch-converted\r\n");
	printf("    tests ./tests/pytorch-operator\r\n");
	printf("    tests ./tests/simple\r\n");
}

int main(int argc, char * argv[])
{
	struct hmap_t * m;
	struct hmap_entry_t * e;
	struct dirent * d;
	struct stat st;
	DIR * dir;

	if(argc != 2)
	{
		usage();
		return -1;
	}
	if((stat(argv[1], &st) != 0) || !S_ISDIR(st.st_mode))
	{
		usage();
		return -1;
	}
	m = hmap_alloc(0);
	if((dir = opendir(argv[1])) != NULL)
	{
		if(chdir(argv[1]) == 0)
		{
			while((d = readdir(dir)) != NULL)
			{
				if((stat(d->d_name, &st) == 0) && S_ISDIR(st.st_mode))
				{
					if(strcmp(".", d->d_name) == 0)
						continue;
					if(strcmp("..", d->d_name) == 0)
						continue;
					hmap_add(m, d->d_name, NULL);
				}
			}
		}
		closedir(dir);
	}
	hmap_sort(m);
	hmap_for_each_entry(e, m)
	{
		testcase(e->key, NULL, 0);
	}
	hmap_free(m, NULL);
	return 0;
}
