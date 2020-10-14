#ifndef __ONNX_H__
#define __ONNX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <hmap.h>
#include <onnx.proto3.pb-c.h>

struct onnx_context_t;
struct onnx_node_t;

struct resolver_t {
	void (*relu)(struct onnx_node_t *);
};

struct onnx_context_t {
	Onnx__ModelProto * model;
	void * buf;
	int buflen;
	struct hmap_t * map;
	struct onnx_node_t * node;
	int nlen;
};

struct onnx_node_t {
	struct onnx_context_t * ctx;
	Onnx__NodeProto * np;
	Onnx__TensorProto ** input;
	int ninput;
	Onnx__TensorProto ** output;
	int noutput;
	void (*operator)(struct onnx_node_t * node);
};

struct onnx_context_t * onnx_context_alloc(const void * buf, int len);
struct onnx_context_t * onnx_context_alloc_from_file(const char * filename);
void onnx_context_free(struct onnx_context_t * ctx);

Onnx__TensorProto * onnx_tensor_alloc(Onnx__ValueInfoProto * v);
void onnx_tensor_free(Onnx__TensorProto * t);
Onnx__TensorProto * onnx_search_tensor(struct onnx_context_t * ctx, const char * name);

void onnx_solve(struct onnx_context_t * ctx);

void onnx_dump_model(struct onnx_context_t * ctx);
void onnx_dump_tensor(Onnx__TensorProto * t);

#ifdef __cplusplus
}
#endif

#endif /* __ONNX_H__ */
