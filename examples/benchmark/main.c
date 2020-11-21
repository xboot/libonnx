#include <sys/time.h>
#include <onnx.h>

struct profiler_t
{
	struct hlist_node node;
	char * name;
	uint64_t begin;
	uint64_t end;
	uint64_t elapsed;
	uint64_t count;
};

#define CONFIG_PROFILER_HASH_SIZE	(4095)
static struct hlist_head __profiler_hash[CONFIG_PROFILER_HASH_SIZE];

static inline uint64_t time_get(void)
{
	struct timeval time;

	gettimeofday(&time, 0);
	return (uint64_t)(time.tv_sec * 1000000000ULL + time.tv_usec * 1000);
}

static inline struct profiler_t * profiler_search(const char * name)
{
	struct profiler_t * p;
	struct hlist_node * n;

	if(!name)
		return NULL;

	hlist_for_each_entry_safe(p, n, &__profiler_hash[shash(name) % CONFIG_PROFILER_HASH_SIZE], node)
	{
		if(strcmp(p->name, name) == 0)
			return p;
	}
	return NULL;
}

static void profiler_init(void)
{
	int i;

	for(i = 0; i < CONFIG_PROFILER_HASH_SIZE; i++)
		init_hlist_head(&__profiler_hash[i]);
}

static void profiler_reset(void)
{
	struct profiler_t * p;
	struct hlist_node * n;
	int i;

	for(i = 0; i < CONFIG_PROFILER_HASH_SIZE; i++)
	{
		hlist_for_each_entry_safe(p, n, &__profiler_hash[i], node)
		{
			hlist_del(&p->node);
			free(p->name);
			free(p);
		}
	}
}

static struct profiler_t * profiler_get(const char * name)
{
	struct profiler_t * p;

	p = profiler_search(name);
	if(!p)
	{
		p = malloc(sizeof(struct profiler_t));
		if(p)
		{
			init_hlist_node(&p->node);
			p->name = strdup(name);
			p->begin = 0;
			p->end = 0;
			p->elapsed = 0;
			p->count = 0;
			hlist_add_head(&p->node, &__profiler_hash[shash(name) % CONFIG_PROFILER_HASH_SIZE]);
		}
	}
	return p;
}

static inline void profiler_begin(struct profiler_t * p)
{
	p->begin = time_get();
}

static inline void profiler_end(struct profiler_t * p)
{
	p->end = time_get();
	p->elapsed += p->end - p->begin;
	p->count++;
}

static void profiler_dump(void)
{
	struct profiler_t * p;
	struct hlist_node * n;
	struct hmap_t * m;
	struct hmap_entry_t * e;
	int i;

	printf("Profiler analysis:\r\n");
	m = hmap_alloc(0);
	for(i = 0; i < CONFIG_PROFILER_HASH_SIZE; i++)
	{
		hlist_for_each_entry_safe(p, n, &__profiler_hash[i], node)
		{
			hmap_add(m, p->name, p);
		}
	}
	hmap_sort(m);
	hmap_for_each_entry(e, m)
	{
		p = (struct profiler_t *)e->value;
	    printf("%-24s %ld %12.3f(us)\r\n", p->name, p->count, (p->count > 0) ? ((double)p->elapsed / 1000.0f) / (double)p->count : 0);
	}
	hmap_free(m, NULL);
}

static void onnx_run_benchmark(struct onnx_context_t * ctx, int count)
{
	struct onnx_node_t * n;
	struct profiler_t * p;
	char name[256];
	int i;

	if(ctx)
	{
		profiler_init();
		while(count-- > 0)
		{
			for(i = 0; i < ctx->nlen; i++)
			{
				n = &ctx->nodes[i];
				sprintf(name, "%s-%d", n->proto->op_type, n->opset);
				p = profiler_get(name);
				profiler_begin(p);
				if(n->reshape(n))
					n->operator(n);
				profiler_end(p);
			}
		}
		profiler_dump();
		profiler_reset();
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
