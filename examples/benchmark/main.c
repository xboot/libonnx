#include <sys/time.h>
#include <onnx.h>

struct profiler_t
{
	uint64_t begin;
	uint64_t end;
	uint64_t elapsed;
	uint64_t count;
};

static inline uint64_t time_get(void)
{
	struct timeval tv;

	gettimeofday(&tv, 0);
	return (uint64_t)(tv.tv_sec * 1000000000ULL + tv.tv_usec * 1000);
}

static struct hmap_t * profiler_alloc(int size)
{
	return hmap_alloc(size);
}

static void hmap_entry_callback(struct hmap_entry_t * e)
{
	if(e && e->value)
		free(e->value);
}

static void profiler_free(struct hmap_t * m)
{
	hmap_free(m, hmap_entry_callback);
}

static struct profiler_t * profiler_search(struct hmap_t * m, const char * name)
{
	struct profiler_t * p = NULL;

	if(m && name)
	{
		p = hmap_search(m, name);
		if(!p)
		{
			p = malloc(sizeof(struct profiler_t));
			if(p)
			{
				p->begin = 0;
				p->end = 0;
				p->elapsed = 0;
				p->count = 0;
				hmap_add(m, name, p);
			}
		}
	}
	return p;
}

static inline void profiler_begin(struct profiler_t * p)
{
	if(p)
		p->begin = time_get();
}

static inline void profiler_end(struct profiler_t * p)
{
	if(p)
	{
		p->end = time_get();
		p->elapsed += p->end - p->begin;
		p->count++;
	}
}

static void profiler_dump(struct hmap_t * m)
{
	struct hmap_entry_t * e;
	struct profiler_t * p;

	if(m)
	{
		printf("Profiler analysis:\r\n");
		hmap_sort(m);
		hmap_for_each_entry(e, m)
		{
			p = (struct profiler_t *)e->value;
			printf("%-32s %ld %12.3f(us)\r\n", e->key, p->count, (p->count > 0) ? ((double)p->elapsed / 1000.0f) / (double)p->count : 0);
		}
	}
}

static void onnx_run_benchmark(struct onnx_context_t * ctx, int count)
{
	struct onnx_node_t * n;
	struct hmap_t * m;
	struct profiler_t * p;
	char name[256];
	int len, i;

	if(ctx)
	{
		m = profiler_alloc(0);
		while(count-- > 0)
		{
			for(i = 0; i < ctx->g->nlen; i++)
			{
				n = &ctx->g->nodes[i];
				len = sprintf(name, "%s-%d", n->proto->op_type, n->opset);
				len += sprintf(name + len, "%*s", 24 - len, n->r->name);
				p = profiler_search(m, name);
				profiler_begin(p);
				if(n->reshape(n))
					n->operator(n);
				profiler_end(p);
			}
		}
		profiler_dump(m);
		profiler_free(m);
	}
}

int main(int argc, char * argv[])
{
	struct onnx_context_t * ctx;
	char * filename = NULL;
	int count = 0;

	if(argc <= 1)
	{
		printf("usage:\r\n");
		printf("    benchmark <filename> [count]\r\n");
		return -1;
	}
	filename = argv[1];
	if(argc >= 3)
		count = strtol(argv[2], NULL, 0);
	if(count <= 0)
		count = 10;

	ctx = onnx_context_alloc_from_file(filename, NULL, 0);
	if(ctx)
	{
		onnx_run_benchmark(ctx, count);
		onnx_context_free(ctx);
	}
	return 0;
}
