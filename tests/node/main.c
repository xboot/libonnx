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

#define FLOAT_EPSILON		(1e-5)

static int onnx_tensor_equal(struct onnx_tensor_t * a, struct onnx_tensor_t * b)
{
	int result = 0;
	int i;

	if(!a || !b)
		return 0;
	if(a->type != b->type)
		return 0;
	if(a->ndim != b->ndim)
		return 0;
	if(a->ndata != b->ndata)
		return 0;
	if(memcmp(a->dims, b->dims, sizeof(int64_t) * a->ndim) != 0)
		return 0;
	switch(a->type)
	{
	case ONNX_TENSOR_TYPE_BOOL:
	case ONNX_TENSOR_TYPE_INT8:
	case ONNX_TENSOR_TYPE_INT16:
	case ONNX_TENSOR_TYPE_INT32:
	case ONNX_TENSOR_TYPE_INT64:
	case ONNX_TENSOR_TYPE_UINT8:
	case ONNX_TENSOR_TYPE_UINT16:
	case ONNX_TENSOR_TYPE_UINT32:
	case ONNX_TENSOR_TYPE_UINT64:
		if(memcmp(a->datas, b->datas, a->ndata * onnx_tensor_type_size(a->type)) == 0)
			result = 1;
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * p = (uint16_t *)a->datas;
			uint16_t * q = (uint16_t *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabsf(bfloat16_to_float32(p[i]) - bfloat16_to_float32(q[i])) > FLOAT_EPSILON)
					break;
			}
			if(i == a->ndata)
				result = 1;
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * p = (uint16_t *)a->datas;
			uint16_t * q = (uint16_t *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabsf(float16_to_float32(p[i]) - float16_to_float32(q[i])) > FLOAT_EPSILON)
					break;
			}
			if(i == a->ndata)
				result = 1;
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * p = (float *)a->datas;
			float * q = (float *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabsf(p[i] - q[i]) > FLOAT_EPSILON)
					break;
			}
			if(i == a->ndata)
				result = 1;
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * p = (double *)a->datas;
			double * q = (double *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabs(p[i] - q[i]) > FLOAT_EPSILON)
					break;
			}
			if(i == a->ndata)
				result = 1;
		}
		break;
	case ONNX_TENSOR_TYPE_COMPLEX64:
		{
			float * p = (float *)a->datas;
			float * q = (float *)b->datas;
			for(i = 0; i < a->ndata * 2; i++)
			{
				if(fabsf(p[i] - q[i]) > FLOAT_EPSILON)
					break;
			}
			if(i == a->ndata * 2)
				result = 1;
		}
		break;
	case ONNX_TENSOR_TYPE_COMPLEX128:
		{
			double * p = (double *)a->datas;
			double * q = (double *)b->datas;
			for(i = 0; i < a->ndata * 2; i++)
			{
				if(fabs(p[i] - q[i]) > FLOAT_EPSILON)
					break;
			}
			if(i == a->ndata * 2)
				result = 1;
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** p = (char **)a->datas;
			char ** q = (char **)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(p[i] && q[i] && (strcmp(p[i], q[i]) != 0))
					break;
			}
			if(i == a->ndata)
				result = 1;
		}
		break;
	default:
		break;
	}
	return result;
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
	int okay;

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
			okay = 0;
			while(1)
			{
				sprintf(tmp, "%s/input_%d.pb", data_set_path, ninput);
				if((lstat(tmp, &st) != 0) || !S_ISREG(st.st_mode))
					break;
				if(ninput > ctx->model->graph->n_input)
					break;
				t = onnx_search_tensor(ctx, ctx->model->graph->input[ninput]->name);
				o = onnx_tensor_alloc_from_file(tmp);
				onnx_tensor_apply(t, o->datas, o->ndata * onnx_tensor_type_size(o->type), &o->scalar);
				onnx_tensor_free(o);
				okay++;
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
				if(onnx_tensor_equal(t, o))
					okay++;
				onnx_tensor_free(o);
				noutput++;
			}

			int len = printf("[%s](test_data_set_%d)", path, data_set_index);
			printf("%*s\r\n", 100 + 12 - 6 - len, ((ninput + noutput == okay) && (okay >= 2)) ? "\033[42;37m[OKAY]\033[0m" : "\033[41;37m[FAIL]\033[0m");
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
			if(strncmp(e->key, "test_clip", strlen("test_clip")) == 0)
				testcase(e->key, r);
		}
		hmap_free(m, NULL);
	}
	return 0;
}
