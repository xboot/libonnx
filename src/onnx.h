#ifndef __ONNX_H__
#define __ONNX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <onnxconf.h>
#include <onnx.proto3.pb-c.h>

#define LIBONNX_MAJOY			(1)
#define LIBONNX_MINIOR			(0)
#define LIBONNX_PATCH			(0)
#define LIBONNX_VERSION			((LIBONNX_MAJOY * 10000) + (LIBONNX_MINIOR * 100) + LIBONNX_PATCH)

struct onnx_tensor_t;
struct onnx_node_t;
struct onnx_graph_t;
struct onnx_context_t;
struct onnx_resolver_t;

enum onnx_tensor_type_t {
	ONNX_TENSOR_TYPE_UNDEFINED		= 0,
	ONNX_TENSOR_TYPE_BOOL			= 9,
	ONNX_TENSOR_TYPE_INT4			= 22,
	ONNX_TENSOR_TYPE_INT8			= 3,
	ONNX_TENSOR_TYPE_INT16			= 5,
	ONNX_TENSOR_TYPE_INT32			= 6,
	ONNX_TENSOR_TYPE_INT64			= 7,
	ONNX_TENSOR_TYPE_UINT4			= 21,
	ONNX_TENSOR_TYPE_UINT8			= 2,
	ONNX_TENSOR_TYPE_UINT16			= 4,
	ONNX_TENSOR_TYPE_UINT32			= 12,
	ONNX_TENSOR_TYPE_UINT64			= 13,
	ONNX_TENSOR_TYPE_FLOAT4E2M1		= 23,
	ONNX_TENSOR_TYPE_FLOAT8E4M3FN	= 17,
	ONNX_TENSOR_TYPE_FLOAT8E4M3FNUZ	= 18,
	ONNX_TENSOR_TYPE_FLOAT8E5M2		= 19,
	ONNX_TENSOR_TYPE_FLOAT8E5M2FNUZ	= 20,
	ONNX_TENSOR_TYPE_FLOAT8E8M0		= 24,
	ONNX_TENSOR_TYPE_BFLOAT16		= 16,
	ONNX_TENSOR_TYPE_FLOAT16		= 10,
	ONNX_TENSOR_TYPE_FLOAT32		= 1,
	ONNX_TENSOR_TYPE_FLOAT64		= 11,
	ONNX_TENSOR_TYPE_COMPLEX64		= 14,
	ONNX_TENSOR_TYPE_COMPLEX128		= 15,
	ONNX_TENSOR_TYPE_STRING			= 8,
};

struct onnx_tensor_t {
	char * name;
	enum onnx_tensor_type_t type;
	int * strides;
	int * dims;
	int ndim;
	void * datas;
	size_t ndata;
};

struct onnx_node_t {
	struct onnx_context_t * ctx;
	struct onnx_resolver_t * r;
	void * rctx;
	int opset;
	struct onnx_tensor_t ** inputs;
	int ninput;
	struct onnx_tensor_t ** outputs;
	int noutput;
	Onnx__NodeProto * proto;

	int (*init)(struct onnx_node_t * n);
	int (*exit)(struct onnx_node_t * n);
	int (*reshape)(struct onnx_node_t * n);
	void (*operator)(struct onnx_node_t * n);
	void * priv;
};

struct onnx_graph_t {
	struct onnx_node_t * nodes;
	int nlen;
};

struct onnx_context_t {
	Onnx__ModelProto * model;
	struct hmap_t * map;
	struct onnx_resolver_t ** r;
	void ** rctx;
	int rlen;
	struct onnx_graph_t * g;
};

struct onnx_resolver_t {
	const char * name;

	void * (*create)(void);
	void (*destroy)(void * rctx);

	/* Operator */
	void (*op_Abs)(struct onnx_node_t * n);
	void (*op_Acos)(struct onnx_node_t * n);
	void (*op_Acosh)(struct onnx_node_t * n);
	void (*op_Add)(struct onnx_node_t * n);
	void (*op_And)(struct onnx_node_t * n);
	void (*op_ArgMax)(struct onnx_node_t * n);
	void (*op_ArgMin)(struct onnx_node_t * n);
	void (*op_Asin)(struct onnx_node_t * n);
	void (*op_Asinh)(struct onnx_node_t * n);
	void (*op_Atan)(struct onnx_node_t * n);
	void (*op_Atanh)(struct onnx_node_t * n);
	void (*op_AveragePool)(struct onnx_node_t * n);
	void (*op_BatchNormalization)(struct onnx_node_t * n);
	void (*op_BitShift)(struct onnx_node_t * n);
	void (*op_BitwiseAnd)(struct onnx_node_t * n);
	void (*op_BitwiseNot)(struct onnx_node_t * n);
	void (*op_BitwiseOr)(struct onnx_node_t * n);
	void (*op_BitwiseXor)(struct onnx_node_t * n);
	void (*op_Cast)(struct onnx_node_t * n);
	void (*op_Ceil)(struct onnx_node_t * n);
	void (*op_Compress)(struct onnx_node_t * n);
	void (*op_Concat)(struct onnx_node_t * n);
	void (*op_ConcatFromSequence)(struct onnx_node_t * n);
	void (*op_Constant)(struct onnx_node_t * n);
	void (*op_ConstantOfShape)(struct onnx_node_t * n);
	void (*op_Conv)(struct onnx_node_t * n);
	void (*op_ConvInteger)(struct onnx_node_t * n);
	void (*op_ConvTranspose)(struct onnx_node_t * n);
	void (*op_Cos)(struct onnx_node_t * n);
	void (*op_Cosh)(struct onnx_node_t * n);
	void (*op_CumSum)(struct onnx_node_t * n);
	void (*op_DepthToSpace)(struct onnx_node_t * n);
	void (*op_DequantizeLinear)(struct onnx_node_t * n);
	void (*op_Det)(struct onnx_node_t * n);
	void (*op_Div)(struct onnx_node_t * n);
	void (*op_Dropout)(struct onnx_node_t * n);
	void (*op_Einsum)(struct onnx_node_t * n);
	void (*op_Equal)(struct onnx_node_t * n);
	void (*op_Erf)(struct onnx_node_t * n);
	void (*op_Exp)(struct onnx_node_t * n);
	void (*op_Expand)(struct onnx_node_t * n);
	void (*op_EyeLike)(struct onnx_node_t * n);
	void (*op_Flatten)(struct onnx_node_t * n);
	void (*op_Floor)(struct onnx_node_t * n);
	void (*op_GRU)(struct onnx_node_t * n);
	void (*op_Gather)(struct onnx_node_t * n);
	void (*op_GatherElements)(struct onnx_node_t * n);
	void (*op_GatherND)(struct onnx_node_t * n);
	void (*op_Gemm)(struct onnx_node_t * n);
	void (*op_GlobalAveragePool)(struct onnx_node_t * n);
	void (*op_GlobalLpPool)(struct onnx_node_t * n);
	void (*op_GlobalMaxPool)(struct onnx_node_t * n);
	void (*op_Greater)(struct onnx_node_t * n);
	void (*op_Hardmax)(struct onnx_node_t * n);
	void (*op_Identity)(struct onnx_node_t * n);
	void (*op_If)(struct onnx_node_t * n);
	void (*op_InstanceNormalization)(struct onnx_node_t * n);
	void (*op_IsInf)(struct onnx_node_t * n);
	void (*op_IsNaN)(struct onnx_node_t * n);
	void (*op_LRN)(struct onnx_node_t * n);
	void (*op_LSTM)(struct onnx_node_t * n);
	void (*op_Less)(struct onnx_node_t * n);
	void (*op_Log)(struct onnx_node_t * n);
	void (*op_Loop)(struct onnx_node_t * n);
	void (*op_LpNormalization)(struct onnx_node_t * n);
	void (*op_LpPool)(struct onnx_node_t * n);
	void (*op_MatMul)(struct onnx_node_t * n);
	void (*op_MatMulInteger)(struct onnx_node_t * n);
	void (*op_Max)(struct onnx_node_t * n);
	void (*op_MaxPool)(struct onnx_node_t * n);
	void (*op_MaxRoiPool)(struct onnx_node_t * n);
	void (*op_MaxUnpool)(struct onnx_node_t * n);
	void (*op_Mean)(struct onnx_node_t * n);
	void (*op_Min)(struct onnx_node_t * n);
	void (*op_Mod)(struct onnx_node_t * n);
	void (*op_Mul)(struct onnx_node_t * n);
	void (*op_Multinomial)(struct onnx_node_t * n);
	void (*op_Neg)(struct onnx_node_t * n);
	void (*op_NonMaxSuppression)(struct onnx_node_t * n);
	void (*op_NonZero)(struct onnx_node_t * n);
	void (*op_Not)(struct onnx_node_t * n);
	void (*op_OneHot)(struct onnx_node_t * n);
	void (*op_Or)(struct onnx_node_t * n);
	void (*op_Pad)(struct onnx_node_t * n);
	void (*op_Pow)(struct onnx_node_t * n);
	void (*op_QLinearConv)(struct onnx_node_t * n);
	void (*op_QLinearMatMul)(struct onnx_node_t * n);
	void (*op_QuantizeLinear)(struct onnx_node_t * n);
	void (*op_RNN)(struct onnx_node_t * n);
	void (*op_RandomNormal)(struct onnx_node_t * n);
	void (*op_RandomNormalLike)(struct onnx_node_t * n);
	void (*op_RandomUniform)(struct onnx_node_t * n);
	void (*op_RandomUniformLike)(struct onnx_node_t * n);
	void (*op_Reciprocal)(struct onnx_node_t * n);
	void (*op_ReduceMax)(struct onnx_node_t * n);
	void (*op_ReduceMean)(struct onnx_node_t * n);
	void (*op_ReduceMin)(struct onnx_node_t * n);
	void (*op_ReduceProd)(struct onnx_node_t * n);
	void (*op_ReduceSum)(struct onnx_node_t * n);
	void (*op_Reshape)(struct onnx_node_t * n);
	void (*op_Resize)(struct onnx_node_t * n);
	void (*op_ReverseSequence)(struct onnx_node_t * n);
	void (*op_RoiAlign)(struct onnx_node_t * n);
	void (*op_Round)(struct onnx_node_t * n);
	void (*op_Scan)(struct onnx_node_t * n);
	void (*op_Scatter)(struct onnx_node_t * n);
	void (*op_ScatterElements)(struct onnx_node_t * n);
	void (*op_ScatterND)(struct onnx_node_t * n);
	void (*op_SequenceAt)(struct onnx_node_t * n);
	void (*op_SequenceConstruct)(struct onnx_node_t * n);
	void (*op_SequenceEmpty)(struct onnx_node_t * n);
	void (*op_SequenceErase)(struct onnx_node_t * n);
	void (*op_SequenceInsert)(struct onnx_node_t * n);
	void (*op_SequenceLength)(struct onnx_node_t * n);
	void (*op_Shape)(struct onnx_node_t * n);
	void (*op_Sigmoid)(struct onnx_node_t * n);
	void (*op_Sign)(struct onnx_node_t * n);
	void (*op_Sin)(struct onnx_node_t * n);
	void (*op_Sinh)(struct onnx_node_t * n);
	void (*op_Size)(struct onnx_node_t * n);
	void (*op_Slice)(struct onnx_node_t * n);
	void (*op_SpaceToDepth)(struct onnx_node_t * n);
	void (*op_Split)(struct onnx_node_t * n);
	void (*op_SplitToSequence)(struct onnx_node_t * n);
	void (*op_Sqrt)(struct onnx_node_t * n);
	void (*op_Squeeze)(struct onnx_node_t * n);
	void (*op_StringNormalizer)(struct onnx_node_t * n);
	void (*op_Sub)(struct onnx_node_t * n);
	void (*op_Sum)(struct onnx_node_t * n);
	void (*op_Tan)(struct onnx_node_t * n);
	void (*op_Tanh)(struct onnx_node_t * n);
	void (*op_TensorScatter)(struct onnx_node_t * n);
	void (*op_TfIdfVectorizer)(struct onnx_node_t * n);
	void (*op_Tile)(struct onnx_node_t * n);
	void (*op_TopK)(struct onnx_node_t * n);
	void (*op_Transpose)(struct onnx_node_t * n);
	void (*op_Trilu)(struct onnx_node_t * n);
	void (*op_Unique)(struct onnx_node_t * n);
	void (*op_Unsqueeze)(struct onnx_node_t * n);
	void (*op_Upsample)(struct onnx_node_t * n);
	void (*op_Where)(struct onnx_node_t * n);
	void (*op_Xor)(struct onnx_node_t * n);

	/* Function */
	void (*op_AffineGrid)(struct onnx_node_t * n);
	void (*op_Attention)(struct onnx_node_t * n);
	void (*op_Bernoulli)(struct onnx_node_t * n);
	void (*op_BlackmanWindow)(struct onnx_node_t * n);
	void (*op_CastLike)(struct onnx_node_t * n);
	void (*op_Celu)(struct onnx_node_t * n);
	void (*op_CenterCropPad)(struct onnx_node_t * n);
	void (*op_Clip)(struct onnx_node_t * n);
	void (*op_DynamicQuantizeLinear)(struct onnx_node_t * n);
	void (*op_Elu)(struct onnx_node_t * n);
	void (*op_Gelu)(struct onnx_node_t * n);
	void (*op_GreaterOrEqual)(struct onnx_node_t * n);
	void (*op_GroupNormalization)(struct onnx_node_t * n);
	void (*op_HammingWindow)(struct onnx_node_t * n);
	void (*op_HannWindow)(struct onnx_node_t * n);
	void (*op_HardSigmoid)(struct onnx_node_t * n);
	void (*op_HardSwish)(struct onnx_node_t * n);
	void (*op_LayerNormalization)(struct onnx_node_t * n);
	void (*op_LeakyRelu)(struct onnx_node_t * n);
	void (*op_LessOrEqual)(struct onnx_node_t * n);
	void (*op_LogSoftmax)(struct onnx_node_t * n);
	void (*op_MeanVarianceNormalization)(struct onnx_node_t * n);
	void (*op_Mish)(struct onnx_node_t * n);
	void (*op_NegativeLogLikelihoodLoss)(struct onnx_node_t * n);
	void (*op_PRelu)(struct onnx_node_t * n);
	void (*op_RMSNormalization)(struct onnx_node_t * n);
	void (*op_Range)(struct onnx_node_t * n);
	void (*op_ReduceL1)(struct onnx_node_t * n);
	void (*op_ReduceL2)(struct onnx_node_t * n);
	void (*op_ReduceLogSum)(struct onnx_node_t * n);
	void (*op_ReduceLogSumExp)(struct onnx_node_t * n);
	void (*op_ReduceSumSquare)(struct onnx_node_t * n);
	void (*op_Relu)(struct onnx_node_t * n);
	void (*op_RotaryEmbedding)(struct onnx_node_t * n);
	void (*op_Selu)(struct onnx_node_t * n);
	void (*op_SequenceMap)(struct onnx_node_t * n);
	void (*op_Shrink)(struct onnx_node_t * n);
	void (*op_Softmax)(struct onnx_node_t * n);
	void (*op_SoftmaxCrossEntropyLoss)(struct onnx_node_t * n);
	void (*op_Softplus)(struct onnx_node_t * n);
	void (*op_Softsign)(struct onnx_node_t * n);
	void (*op_Swish)(struct onnx_node_t * n);
	void (*op_ThresholdedRelu)(struct onnx_node_t * n);
};

struct onnx_context_t * onnx_context_alloc(const void * buf, size_t len, struct onnx_resolver_t ** r, int rlen);
struct onnx_context_t * onnx_context_alloc_from_file(const char * filename, struct onnx_resolver_t ** r, int rlen);
void onnx_context_free(struct onnx_context_t * ctx);

struct onnx_graph_t * onnx_graph_alloc(struct onnx_context_t * ctx, Onnx__GraphProto * graph);
void onnx_graph_free(struct onnx_graph_t * g);

const char * onnx_tensor_type_tostring(enum onnx_tensor_type_t type);
int onnx_tensor_type_sizeof(enum onnx_tensor_type_t type);
struct onnx_tensor_t * onnx_tensor_search(struct onnx_context_t * ctx, const char * name);
struct onnx_tensor_t * onnx_tensor_alloc(const char * name, enum onnx_tensor_type_t type, int * dims, int ndim);
struct onnx_tensor_t * onnx_tensor_alloc_from_file(const char * filename);
void onnx_tensor_free(struct onnx_tensor_t * t);
int onnx_tensor_equal(struct onnx_tensor_t * a, struct onnx_tensor_t * b);
void onnx_tensor_reinit(struct onnx_tensor_t * t, enum onnx_tensor_type_t type, int * dims, int ndim);
void onnx_tensor_apply(struct onnx_tensor_t * t, void * buf, size_t len);

static inline int onnx_tensor_is_scalar(struct onnx_tensor_t * t)
{
	return ((t->ndim == 0) && (t->ndata == 1)) ? 1 : 0;
}

static inline int onnx_tensor_broadcast_is_valid(struct onnx_tensor_t * x, int * dims, int ndim)
{
	int i;

	if(x->ndim > ndim)
		return 0;
	for(i = 1; i <= x->ndim; i++)
	{
		if((x->dims[x->ndim - i] != 1) && (x->dims[x->ndim - i] != dims[ndim - i]))
			return 0;
	}
	return 1;
}

static inline int onnx_tensor_indices_to_offset(struct onnx_tensor_t * t, int * indices)
{
	int offset, i;

	for(i = 0, offset = 0; i < t->ndim; i++)
		offset += indices[i] * t->strides[i];
	return offset;
}

static inline void onnx_tensor_offset_to_indices(struct onnx_tensor_t * t, int offset, int * indices)
{
	int i;

	for(i = t->ndim - 1; i >= 0; i--)
	{
		indices[i] = offset % t->dims[i];
		offset /= t->dims[i];
	}
}

static inline int onnx_tensor_reshape(struct onnx_tensor_t * y, int * dims, int ndim, enum onnx_tensor_type_t type)
{
	if((y->ndim != ndim) || (dims && (onnx_memcmp(y->dims, dims, sizeof(int) * y->ndim) != 0)) || (y->type != type))
		onnx_tensor_reinit(y, type, dims, ndim);
	return 1;
}

static inline int onnx_tensor_reshape_identity(struct onnx_tensor_t * y, struct onnx_tensor_t * x, enum onnx_tensor_type_t type)
{
	if((y->ndim != x->ndim) || (onnx_memcmp(y->dims, x->dims, sizeof(int) * y->ndim) != 0) || (y->type != type))
		onnx_tensor_reinit(y, type, x->dims, x->ndim);
	return 1;
}

static inline int onnx_tensor_reshape_multi_broadcast(struct onnx_tensor_t * y, struct onnx_tensor_t * a, struct onnx_tensor_t * b, enum onnx_tensor_type_t type)
{
	int ndim = XMAX(a->ndim, b->ndim);
	int dims[ndim];
	int i, j, k;

	if(ndim > 0)
	{
		for(i = a->ndim - 1, j = b->ndim - 1, k = ndim - 1; k >= 0; k--)
		{
			if(i < 0)
				dims[k] = b->dims[j--];
			else if(j < 0)
				dims[k] = a->dims[i--];
			else
			{
				if(a->dims[i] == b->dims[j])
					dims[k] = a->dims[i];
				else if((a->dims[i] == 1) || (b->dims[j] == 1))
					dims[k] = (a->dims[i] > b->dims[j]) ? a->dims[i] : b->dims[j];
				else
					return 0;
				i--;
				j--;
			}
		}
	}
	if((y->type != type) || (y->ndim != ndim) || (onnx_memcmp(y->dims, dims, sizeof(int) * ndim) != 0))
		onnx_tensor_reinit(y, type, dims, ndim);
	return 1;
}

static inline void * onnx_tensor_broadcast_map_address(struct onnx_tensor_t * x, struct onnx_tensor_t * y, int offset)
{
	int xndim = x->ndim;
	int yndim = y->ndim;

	if((xndim > 0) && (yndim > 0))
	{
		int dndim = yndim - xndim;
		int ix[xndim];
		int iy[yndim];
		int i;

		onnx_tensor_offset_to_indices(y, offset, iy);
		for(i = 0; i < xndim; i++)
			ix[i] = iy[dndim + i] % x->dims[i];
		return x->datas + onnx_tensor_indices_to_offset(x, ix) * onnx_tensor_type_sizeof(x->type);
	}
	return x->datas;
}

float onnx_attribute_read_float(struct onnx_node_t * n, const char * name, float def);
int64_t onnx_attribute_read_int(struct onnx_node_t * n, const char * name, int64_t def);
char * onnx_attribute_read_string(struct onnx_node_t * n, const char * name, char * def);
int onnx_attribute_read_ints(struct onnx_node_t * n, const char * name, int64_t ** ints);
int onnx_attribute_read_floats(struct onnx_node_t * n, const char * name, float ** floats);
int onnx_attribute_read_tensor(struct onnx_node_t * n, const char * name, struct onnx_tensor_t * t);
Onnx__GraphProto * onnx_attribute_read_graph(struct onnx_node_t * n, const char * name, Onnx__GraphProto * def);
Onnx__SparseTensorProto * onnx_attribute_read_sparse_tensor(struct onnx_node_t * n, const char * name, Onnx__SparseTensorProto * def);

void onnx_tensor_dump(struct onnx_tensor_t * t, int detail);
void onnx_node_dump(struct onnx_node_t * n, int detail);
void onnx_graph_dump(struct onnx_graph_t * g, int detail);
void onnx_context_dump(struct onnx_context_t * ctx, int detail);

void onnx_run(struct onnx_context_t * ctx);

#ifdef __cplusplus
}
#endif

#endif /* __ONNX_H__ */
