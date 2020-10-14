#ifndef __DUMP_H__
#define __DUMP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <onnx.h>

struct onnx_context_t;
struct onnx_node_t;
struct resolver_t;

void onnx_dump_node(struct onnx_node_t * n);
void onnx_dump_model(struct onnx_context_t * ctx);

#ifdef __cplusplus
}
#endif

#endif /* __DUMP_H__ */
