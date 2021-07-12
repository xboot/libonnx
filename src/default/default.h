#ifndef __DEFAULT_H__
#define __DEFAULT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <onnx.h>

void * resolver_default_create(void);
void resolver_default_destroy(void * rctx);

void resolver_default_op_Abs(struct onnx_node_t * n);
void resolver_default_op_Acos(struct onnx_node_t * n);
void resolver_default_op_Acosh(struct onnx_node_t * n);
void resolver_default_op_Add(struct onnx_node_t * n);
void resolver_default_op_And(struct onnx_node_t * n);
void resolver_default_op_ArgMax(struct onnx_node_t * n);
void resolver_default_op_ArgMin(struct onnx_node_t * n);
void resolver_default_op_Asin(struct onnx_node_t * n);
void resolver_default_op_Asinh(struct onnx_node_t * n);
void resolver_default_op_Atan(struct onnx_node_t * n);
void resolver_default_op_Atanh(struct onnx_node_t * n);
void resolver_default_op_AveragePool(struct onnx_node_t * n);
void resolver_default_op_BatchNormalization(struct onnx_node_t * n);
void resolver_default_op_BitShift(struct onnx_node_t * n);
void resolver_default_op_Cast(struct onnx_node_t * n);
void resolver_default_op_Ceil(struct onnx_node_t * n);
void resolver_default_op_Clip(struct onnx_node_t * n);
void resolver_default_op_Compress(struct onnx_node_t * n);
void resolver_default_op_Concat(struct onnx_node_t * n);
void resolver_default_op_ConcatFromSequence(struct onnx_node_t * n);
void resolver_default_op_Constant(struct onnx_node_t * n);
void resolver_default_op_ConstantOfShape(struct onnx_node_t * n);
void resolver_default_op_Conv(struct onnx_node_t * n);
void resolver_default_op_ConvInteger(struct onnx_node_t * n);
void resolver_default_op_ConvTranspose(struct onnx_node_t * n);
void resolver_default_op_Cos(struct onnx_node_t * n);
void resolver_default_op_Cosh(struct onnx_node_t * n);
void resolver_default_op_CumSum(struct onnx_node_t * n);
void resolver_default_op_DepthToSpace(struct onnx_node_t * n);
void resolver_default_op_DequantizeLinear(struct onnx_node_t * n);
void resolver_default_op_Det(struct onnx_node_t * n);
void resolver_default_op_Div(struct onnx_node_t * n);
void resolver_default_op_Dropout(struct onnx_node_t * n);
void resolver_default_op_Einsum(struct onnx_node_t * n);
void resolver_default_op_Elu(struct onnx_node_t * n);
void resolver_default_op_Equal(struct onnx_node_t * n);
void resolver_default_op_Erf(struct onnx_node_t * n);
void resolver_default_op_Exp(struct onnx_node_t * n);
void resolver_default_op_Expand(struct onnx_node_t * n);
void resolver_default_op_EyeLike(struct onnx_node_t * n);
void resolver_default_op_Flatten(struct onnx_node_t * n);
void resolver_default_op_Floor(struct onnx_node_t * n);
void resolver_default_op_GRU(struct onnx_node_t * n);
void resolver_default_op_Gather(struct onnx_node_t * n);
void resolver_default_op_GatherElements(struct onnx_node_t * n);
void resolver_default_op_GatherND(struct onnx_node_t * n);
void resolver_default_op_Gemm(struct onnx_node_t * n);
void resolver_default_op_GlobalAveragePool(struct onnx_node_t * n);
void resolver_default_op_GlobalLpPool(struct onnx_node_t * n);
void resolver_default_op_GlobalMaxPool(struct onnx_node_t * n);
void resolver_default_op_Greater(struct onnx_node_t * n);
void resolver_default_op_HardSigmoid(struct onnx_node_t * n);
void resolver_default_op_Hardmax(struct onnx_node_t * n);
void resolver_default_op_Identity(struct onnx_node_t * n);
void resolver_default_op_If(struct onnx_node_t * n);
void resolver_default_op_InstanceNormalization(struct onnx_node_t * n);
void resolver_default_op_IsInf(struct onnx_node_t * n);
void resolver_default_op_IsNaN(struct onnx_node_t * n);
void resolver_default_op_LRN(struct onnx_node_t * n);
void resolver_default_op_LSTM(struct onnx_node_t * n);
void resolver_default_op_LeakyRelu(struct onnx_node_t * n);
void resolver_default_op_Less(struct onnx_node_t * n);
void resolver_default_op_Log(struct onnx_node_t * n);
void resolver_default_op_Loop(struct onnx_node_t * n);
void resolver_default_op_LpNormalization(struct onnx_node_t * n);
void resolver_default_op_LpPool(struct onnx_node_t * n);
void resolver_default_op_MatMul(struct onnx_node_t * n);
void resolver_default_op_MatMulInteger(struct onnx_node_t * n);
void resolver_default_op_Max(struct onnx_node_t * n);
void resolver_default_op_MaxPool(struct onnx_node_t * n);
void resolver_default_op_MaxRoiPool(struct onnx_node_t * n);
void resolver_default_op_MaxUnpool(struct onnx_node_t * n);
void resolver_default_op_Mean(struct onnx_node_t * n);
void resolver_default_op_Min(struct onnx_node_t * n);
void resolver_default_op_Mod(struct onnx_node_t * n);
void resolver_default_op_Mul(struct onnx_node_t * n);
void resolver_default_op_Multinomial(struct onnx_node_t * n);
void resolver_default_op_Neg(struct onnx_node_t * n);
void resolver_default_op_NonMaxSuppression(struct onnx_node_t * n);
void resolver_default_op_NonZero(struct onnx_node_t * n);
void resolver_default_op_Not(struct onnx_node_t * n);
void resolver_default_op_OneHot(struct onnx_node_t * n);
void resolver_default_op_Or(struct onnx_node_t * n);
void resolver_default_op_PRelu(struct onnx_node_t * n);
void resolver_default_op_Pad(struct onnx_node_t * n);
void resolver_default_op_Pow(struct onnx_node_t * n);
void resolver_default_op_QLinearConv(struct onnx_node_t * n);
void resolver_default_op_QLinearMatMul(struct onnx_node_t * n);
void resolver_default_op_QuantizeLinear(struct onnx_node_t * n);
void resolver_default_op_RNN(struct onnx_node_t * n);
void resolver_default_op_RandomNormal(struct onnx_node_t * n);
void resolver_default_op_RandomNormalLike(struct onnx_node_t * n);
void resolver_default_op_RandomUniform(struct onnx_node_t * n);
void resolver_default_op_RandomUniformLike(struct onnx_node_t * n);
void resolver_default_op_Reciprocal(struct onnx_node_t * n);
void resolver_default_op_ReduceL1(struct onnx_node_t * n);
void resolver_default_op_ReduceL2(struct onnx_node_t * n);
void resolver_default_op_ReduceLogSum(struct onnx_node_t * n);
void resolver_default_op_ReduceLogSumExp(struct onnx_node_t * n);
void resolver_default_op_ReduceMax(struct onnx_node_t * n);
void resolver_default_op_ReduceMean(struct onnx_node_t * n);
void resolver_default_op_ReduceMin(struct onnx_node_t * n);
void resolver_default_op_ReduceProd(struct onnx_node_t * n);
void resolver_default_op_ReduceSum(struct onnx_node_t * n);
void resolver_default_op_ReduceSumSquare(struct onnx_node_t * n);
void resolver_default_op_Relu(struct onnx_node_t * n);
void resolver_default_op_Reshape(struct onnx_node_t * n);
void resolver_default_op_Resize(struct onnx_node_t * n);
void resolver_default_op_ReverseSequence(struct onnx_node_t * n);
void resolver_default_op_RoiAlign(struct onnx_node_t * n);
void resolver_default_op_Round(struct onnx_node_t * n);
void resolver_default_op_Scan(struct onnx_node_t * n);
void resolver_default_op_Scatter(struct onnx_node_t * n);
void resolver_default_op_ScatterElements(struct onnx_node_t * n);
void resolver_default_op_ScatterND(struct onnx_node_t * n);
void resolver_default_op_Selu(struct onnx_node_t * n);
void resolver_default_op_SequenceAt(struct onnx_node_t * n);
void resolver_default_op_SequenceConstruct(struct onnx_node_t * n);
void resolver_default_op_SequenceEmpty(struct onnx_node_t * n);
void resolver_default_op_SequenceErase(struct onnx_node_t * n);
void resolver_default_op_SequenceInsert(struct onnx_node_t * n);
void resolver_default_op_SequenceLength(struct onnx_node_t * n);
void resolver_default_op_Shape(struct onnx_node_t * n);
void resolver_default_op_Shrink(struct onnx_node_t * n);
void resolver_default_op_Sigmoid(struct onnx_node_t * n);
void resolver_default_op_Sign(struct onnx_node_t * n);
void resolver_default_op_Sin(struct onnx_node_t * n);
void resolver_default_op_Sinh(struct onnx_node_t * n);
void resolver_default_op_Size(struct onnx_node_t * n);
void resolver_default_op_Slice(struct onnx_node_t * n);
void resolver_default_op_Softplus(struct onnx_node_t * n);
void resolver_default_op_Softsign(struct onnx_node_t * n);
void resolver_default_op_SpaceToDepth(struct onnx_node_t * n);
void resolver_default_op_Split(struct onnx_node_t * n);
void resolver_default_op_SplitToSequence(struct onnx_node_t * n);
void resolver_default_op_Sqrt(struct onnx_node_t * n);
void resolver_default_op_Squeeze(struct onnx_node_t * n);
void resolver_default_op_StringNormalizer(struct onnx_node_t * n);
void resolver_default_op_Sub(struct onnx_node_t * n);
void resolver_default_op_Sum(struct onnx_node_t * n);
void resolver_default_op_Tan(struct onnx_node_t * n);
void resolver_default_op_Tanh(struct onnx_node_t * n);
void resolver_default_op_TfIdfVectorizer(struct onnx_node_t * n);
void resolver_default_op_ThresholdedRelu(struct onnx_node_t * n);
void resolver_default_op_Tile(struct onnx_node_t * n);
void resolver_default_op_TopK(struct onnx_node_t * n);
void resolver_default_op_Transpose(struct onnx_node_t * n);
void resolver_default_op_Trilu(struct onnx_node_t * n);
void resolver_default_op_Unique(struct onnx_node_t * n);
void resolver_default_op_Unsqueeze(struct onnx_node_t * n);
void resolver_default_op_Upsample(struct onnx_node_t * n);
void resolver_default_op_Where(struct onnx_node_t * n);
void resolver_default_op_Xor(struct onnx_node_t * n);

void resolver_default_op_Celu(struct onnx_node_t * n);
void resolver_default_op_DynamicQuantizeLinear(struct onnx_node_t * n);
void resolver_default_op_GreaterOrEqual(struct onnx_node_t * n);
void resolver_default_op_HardSwish(struct onnx_node_t * n);
void resolver_default_op_LessOrEqual(struct onnx_node_t * n);
void resolver_default_op_LogSoftmax(struct onnx_node_t * n);
void resolver_default_op_MeanVarianceNormalization(struct onnx_node_t * n);
void resolver_default_op_NegativeLogLikelihoodLoss(struct onnx_node_t * n);
void resolver_default_op_Range(struct onnx_node_t * n);
void resolver_default_op_Softmax(struct onnx_node_t * n);
void resolver_default_op_SoftmaxCrossEntropyLoss(struct onnx_node_t * n);

extern struct onnx_resolver_t resolver_default;

#ifdef __cplusplus
}
#endif

#endif /* __DEFAULT_H__ */
