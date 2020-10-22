/*
 * onnx.c
 *
 * Copyright(c) 2007-2020 Jianjun Jiang <8192542@qq.com>
 * Mobile phone: +86-18665388956
 * QQ: 8192542
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include <onnx.h>

#define ONNX_LOG(...)	printf(__VA_ARGS__)

static struct resolver_t default_resolver = {
	.name 							= "default",

	.op_Abs							= default_resolver_op_Abs,
	.op_Acos						= default_resolver_op_Acos,
	.op_Acosh						= default_resolver_op_Acosh,
	.op_Add							= default_resolver_op_Add,
	.op_And							= default_resolver_op_And,
	.op_ArgMax						= default_resolver_op_ArgMax,
	.op_ArgMin						= default_resolver_op_ArgMin,
	.op_Asin						= default_resolver_op_Asin,
	.op_Asinh						= default_resolver_op_Asinh,
	.op_Atan						= default_resolver_op_Atan,
	.op_Atanh						= default_resolver_op_Atanh,
	.op_AveragePool					= default_resolver_op_AveragePool,
	.op_BatchNormalization			= default_resolver_op_BatchNormalization,
	.op_BitShift					= default_resolver_op_BitShift,
	.op_Cast						= default_resolver_op_Cast,
	.op_Ceil						= default_resolver_op_Ceil,
	.op_Clip						= default_resolver_op_Clip,
	.op_Compress					= default_resolver_op_Compress,
	.op_Concat						= default_resolver_op_Concat,
	.op_ConcatFromSequence			= default_resolver_op_ConcatFromSequence,
	.op_Constant					= default_resolver_op_Constant,
	.op_ConstantOfShape				= default_resolver_op_ConstantOfShape,
	.op_Conv						= default_resolver_op_Conv,
	.op_ConvInteger					= default_resolver_op_ConvInteger,
	.op_ConvTranspose				= default_resolver_op_ConvTranspose,
	.op_Cos							= default_resolver_op_Cos,
	.op_Cosh						= default_resolver_op_Cosh,
	.op_CumSum						= default_resolver_op_CumSum,
	.op_DepthToSpace				= default_resolver_op_DepthToSpace,
	.op_DequantizeLinear			= default_resolver_op_DequantizeLinear,
	.op_Det							= default_resolver_op_Det,
	.op_Div							= default_resolver_op_Div,
	.op_Dropout						= default_resolver_op_Dropout,
	.op_Einsum						= default_resolver_op_Einsum,
	.op_Elu							= default_resolver_op_Elu,
	.op_Equal						= default_resolver_op_Equal,
	.op_Erf							= default_resolver_op_Erf,
	.op_Exp							= default_resolver_op_Exp,
	.op_Expand						= default_resolver_op_Expand,
	.op_EyeLike						= default_resolver_op_EyeLike,
	.op_Flatten						= default_resolver_op_Flatten,
	.op_Floor						= default_resolver_op_Floor,
	.op_GRU							= default_resolver_op_GRU,
	.op_Gather						= default_resolver_op_Gather,
	.op_GatherElements				= default_resolver_op_GatherElements,
	.op_GatherND					= default_resolver_op_GatherND,
	.op_Gemm						= default_resolver_op_Gemm,
	.op_GlobalAveragePool			= default_resolver_op_GlobalAveragePool,
	.op_GlobalLpPool				= default_resolver_op_GlobalLpPool,
	.op_GlobalMaxPool				= default_resolver_op_GlobalMaxPool,
	.op_Greater						= default_resolver_op_Greater,
	.op_HardSigmoid					= default_resolver_op_HardSigmoid,
	.op_Hardmax						= default_resolver_op_Hardmax,
	.op_Identity					= default_resolver_op_Identity,
	.op_If							= default_resolver_op_If,
	.op_InstanceNormalization		= default_resolver_op_InstanceNormalization,
	.op_IsInf						= default_resolver_op_IsInf,
	.op_IsNaN						= default_resolver_op_IsNaN,
	.op_LRN							= default_resolver_op_LRN,
	.op_LSTM						= default_resolver_op_LSTM,
	.op_LeakyRelu					= default_resolver_op_LeakyRelu,
	.op_Less						= default_resolver_op_Less,
	.op_Log							= default_resolver_op_Log,
	.op_Loop						= default_resolver_op_Loop,
	.op_LpNormalization				= default_resolver_op_LpNormalization,
	.op_LpPool						= default_resolver_op_LpPool,
	.op_MatMul						= default_resolver_op_MatMul,
	.op_MatMulInteger				= default_resolver_op_MatMulInteger,
	.op_Max							= default_resolver_op_Max,
	.op_MaxPool						= default_resolver_op_MaxPool,
	.op_MaxRoiPool					= default_resolver_op_MaxRoiPool,
	.op_MaxUnpool					= default_resolver_op_MaxUnpool,
	.op_Mean						= default_resolver_op_Mean,
	.op_Min							= default_resolver_op_Min,
	.op_Mod							= default_resolver_op_Mod,
	.op_Mul							= default_resolver_op_Mul,
	.op_Multinomial					= default_resolver_op_Multinomial,
	.op_Neg							= default_resolver_op_Neg,
	.op_NonMaxSuppression			= default_resolver_op_NonMaxSuppression,
	.op_NonZero						= default_resolver_op_NonZero,
	.op_Not							= default_resolver_op_Not,
	.op_OneHot						= default_resolver_op_OneHot,
	.op_Or							= default_resolver_op_Or,
	.op_PRelu						= default_resolver_op_PRelu,
	.op_Pad							= default_resolver_op_Pad,
	.op_Pow							= default_resolver_op_Pow,
	.op_QLinearConv					= default_resolver_op_QLinearConv,
	.op_QLinearMatMul				= default_resolver_op_QLinearMatMul,
	.op_QuantizeLinear				= default_resolver_op_QuantizeLinear,
	.op_RNN							= default_resolver_op_RNN,
	.op_RandomNormal				= default_resolver_op_RandomNormal,
	.op_RandomNormalLike			= default_resolver_op_RandomNormalLike,
	.op_RandomUniform				= default_resolver_op_RandomUniform,
	.op_RandomUniformLike			= default_resolver_op_RandomUniformLike,
	.op_Reciprocal					= default_resolver_op_Reciprocal,
	.op_ReduceL1					= default_resolver_op_ReduceL1,
	.op_ReduceL2					= default_resolver_op_ReduceL2,
	.op_ReduceLogSum				= default_resolver_op_ReduceLogSum,
	.op_ReduceLogSumExp				= default_resolver_op_ReduceLogSumExp,
	.op_ReduceMax					= default_resolver_op_ReduceMax,
	.op_ReduceMean					= default_resolver_op_ReduceMean,
	.op_ReduceMin					= default_resolver_op_ReduceMin,
	.op_ReduceProd					= default_resolver_op_ReduceProd,
	.op_ReduceSum					= default_resolver_op_ReduceSum,
	.op_ReduceSumSquare				= default_resolver_op_ReduceSumSquare,
	.op_Relu						= default_resolver_op_Relu,
	.op_Reshape						= default_resolver_op_Reshape,
	.op_Resize						= default_resolver_op_Resize,
	.op_ReverseSequence				= default_resolver_op_ReverseSequence,
	.op_RoiAlign					= default_resolver_op_RoiAlign,
	.op_Round						= default_resolver_op_Round,
	.op_Scan						= default_resolver_op_Scan,
	.op_ScatterElements				= default_resolver_op_ScatterElements,
	.op_ScatterND					= default_resolver_op_ScatterND,
	.op_Selu						= default_resolver_op_Selu,
	.op_SequenceAt					= default_resolver_op_SequenceAt,
	.op_SequenceConstruct			= default_resolver_op_SequenceConstruct,
	.op_SequenceEmpty				= default_resolver_op_SequenceEmpty,
	.op_SequenceErase				= default_resolver_op_SequenceErase,
	.op_SequenceInsert				= default_resolver_op_SequenceInsert,
	.op_SequenceLength				= default_resolver_op_SequenceLength,
	.op_Shape						= default_resolver_op_Shape,
	.op_Shrink						= default_resolver_op_Shrink,
	.op_Sigmoid						= default_resolver_op_Sigmoid,
	.op_Sign						= default_resolver_op_Sign,
	.op_Sin							= default_resolver_op_Sin,
	.op_Sinh						= default_resolver_op_Sinh,
	.op_Size						= default_resolver_op_Size,
	.op_Slice						= default_resolver_op_Slice,
	.op_Softplus					= default_resolver_op_Softplus,
	.op_Softsign					= default_resolver_op_Softsign,
	.op_SpaceToDepth				= default_resolver_op_SpaceToDepth,
	.op_Split						= default_resolver_op_Split,
	.op_SplitToSequence				= default_resolver_op_SplitToSequence,
	.op_Sqrt						= default_resolver_op_Sqrt,
	.op_Squeeze						= default_resolver_op_Squeeze,
	.op_StringNormalizer			= default_resolver_op_StringNormalizer,
	.op_Sub							= default_resolver_op_Sub,
	.op_Sum							= default_resolver_op_Sum,
	.op_Tan							= default_resolver_op_Tan,
	.op_Tanh						= default_resolver_op_Tanh,
	.op_TfIdfVectorizer				= default_resolver_op_TfIdfVectorizer,
	.op_ThresholdedRelu				= default_resolver_op_ThresholdedRelu,
	.op_Tile						= default_resolver_op_Tile,
	.op_TopK						= default_resolver_op_TopK,
	.op_Transpose					= default_resolver_op_Transpose,
	.op_Unique						= default_resolver_op_Unique,
	.op_Unsqueeze					= default_resolver_op_Unsqueeze,
	.op_Where						= default_resolver_op_Where,
	.op_Xor							= default_resolver_op_Xor,

	.op_Celu						= default_resolver_op_Celu,
	.op_DynamicQuantizeLinear		= default_resolver_op_DynamicQuantizeLinear,
	.op_GreaterOrEqual				= default_resolver_op_GreaterOrEqual,
	.op_LessOrEqual					= default_resolver_op_LessOrEqual,
	.op_LogSoftmax					= default_resolver_op_LogSoftmax,
	.op_MeanVarianceNormalization	= default_resolver_op_MeanVarianceNormalization,
	.op_NegativeLogLikelihoodLoss	= default_resolver_op_NegativeLogLikelihoodLoss,
	.op_Range						= default_resolver_op_Range,
	.op_Softmax						= default_resolver_op_Softmax,
	.op_SoftmaxCrossEntropyLoss		= default_resolver_op_SoftmaxCrossEntropyLoss,
};

static void resolver_solve_operator(struct resolver_t * r, struct onnx_node_t * n)
{
	void (*rop)(struct onnx_node_t *);

	if(r && n)
	{
		switch(shash(n->proto->op_type))
		{
		case 0x0b87d47b: /* "Abs" */
			rop = r->op_Abs;
			break;
		case 0x7c82680b: /* "Acos" */
			rop = r->op_Acos;
			break;
		case 0x0ccf69d3: /* "Acosh" */
			rop = r->op_Acosh;
			break;
		case 0x0b87d4ae: /* "Add" */
			rop = r->op_Add;
			break;
		case 0x0b87d5f8: /* "And" */
			rop = r->op_And;
			break;
		case 0xa7c70ea5: /* "ArgMax" */
			rop = r->op_ArgMax;
			break;
		case 0xa7c70fa3: /* "ArgMin" */
			rop = r->op_ArgMin;
			break;
		case 0x7c82ab50: /* "Asin" */
			rop = r->op_Asin;
			break;
		case 0x0cd815b8: /* "Asinh" */
			rop = r->op_Asinh;
			break;
		case 0x7c82ae89: /* "Atan" */
			rop = r->op_Atan;
			break;
		case 0x0cd88011: /* "Atanh" */
			rop = r->op_Atanh;
			break;
		case 0xf1a1e23a: /* "AveragePool" */
			rop = r->op_AveragePool;
			break;
		case 0x2d3b46ee: /* "BatchNormalization" */
			rop = r->op_BatchNormalization;
			break;
		case 0x0bfe45a2: /* "BitShift" */
			rop = r->op_BitShift;
			break;
		case 0x7c8378d0: /* "Cast" */
			rop = r->op_Cast;
			break;
		case 0x7c838882: /* "Ceil" */
			rop = r->op_Ceil;
			break;
		case 0x7c83a64d: /* "Clip" */
			rop = r->op_Clip;
			break;
		case 0xb7db9db1: /* "Compress" */
			rop = r->op_Compress;
			break;
		case 0xac3f4a9d: /* "Concat" */
			rop = r->op_Concat;
			break;
		case 0x5053caca: /* "ConcatFromSequence" */
			rop = r->op_ConcatFromSequence;
			break;
		case 0xba6816ef: /* "Constant" */
			rop = r->op_Constant;
			break;
		case 0xe468a875: /* "ConstantOfShape" */
			rop = r->op_ConstantOfShape;
			break;
		case 0x7c83b3bb: /* "Conv" */
			rop = r->op_Conv;
			break;
		case 0x8371dbe9: /* "ConvInteger" */
			rop = r->op_ConvInteger;
			break;
		case 0x3903c4ba: /* "ConvTranspose" */
			rop = r->op_ConvTranspose;
			break;
		case 0x0b87deaa: /* "Cos" */
			rop = r->op_Cos;
			break;
		case 0x7c83b452: /* "Cosh" */
			rop = r->op_Cosh;
			break;
		case 0xacab0fbf: /* "CumSum" */
			rop = r->op_CumSum;
			break;
		case 0xc9c1d669: /* "DepthToSpace" */
			rop = r->op_DepthToSpace;
			break;
		case 0xf9cc985a: /* "DequantizeLinear" */
			rop = r->op_DequantizeLinear;
			break;
		case 0x0b87e1a2: /* "Det" */
			rop = r->op_Det;
			break;
		case 0x0b87e228: /* "Div" */
			rop = r->op_Div;
			break;
		case 0x883bca72: /* "Dropout" */
			rop = r->op_Dropout;
			break;
		case 0xb07d4f76: /* "Einsum" */
			rop = r->op_Einsum;
			break;
		case 0x0b87e6cb: /* "Elu" */
			rop = r->op_Elu;
			break;
		case 0x0d1f905d: /* "Equal" */
			rop = r->op_Equal;
			break;
		case 0x0b87e782: /* "Erf" */
			rop = r->op_Erf;
			break;
		case 0x0b87e852: /* "Exp" */
			rop = r->op_Exp;
			break;
		case 0xb18d8a45: /* "Expand" */
			rop = r->op_Expand;
			break;
		case 0xe4c1560d: /* "EyeLike" */
			rop = r->op_EyeLike;
			break;
		case 0x13363dd3: /* "Flatten" */
			rop = r->op_Flatten;
			break;
		case 0x0d2ed347: /* "Floor" */
			rop = r->op_Floor;
			break;
		case 0x0b87ebd3: /* "GRU" */
			rop = r->op_GRU;
			break;
		case 0xb499f620: /* "Gather" */
			rop = r->op_Gather;
			break;
		case 0x7c94d43d: /* "GatherElements" */
			rop = r->op_GatherElements;
			break;
		case 0x42f00872: /* "GatherND" */
			rop = r->op_GatherND;
			break;
		case 0x7c85ba8b: /* "Gemm" */
			rop = r->op_Gemm;
			break;
		case 0x9289c84b: /* "GlobalAveragePool" */
			rop = r->op_GlobalAveragePool;
			break;
		case 0x3f5a29ac: /* "GlobalLpPool" */
			rop = r->op_GlobalLpPool;
			break;
		case 0x575f0fb6: /* "GlobalMaxPool" */
			rop = r->op_GlobalMaxPool;
			break;
		case 0x6e6d652f: /* "Greater" */
			rop = r->op_Greater;
			break;
		case 0x10341df0: /* "HardSigmoid" */
			rop = r->op_HardSigmoid;
			break;
		case 0x94acb4aa: /* "Hardmax" */
			rop = r->op_Hardmax;
			break;
		case 0xdfd9b28f: /* "Identity" */
			rop = r->op_Identity;
			break;
		case 0x00597414: /* "If" */
			rop = r->op_If;
			break;
		case 0xfb0902c1: /* "InstanceNormalization" */
			rop = r->op_InstanceNormalization;
			break;
		case 0x0d68519e: /* "IsInf" */
			rop = r->op_IsInf;
			break;
		case 0x0d68651e: /* "IsNaN" */
			rop = r->op_IsNaN;
			break;
		case 0x0b880111: /* "LRN" */
			rop = r->op_LRN;
			break;
		case 0x7c882885: /* "LSTM" */
			rop = r->op_LSTM;
			break;
		case 0xea2c5c33: /* "LeakyRelu" */
			rop = r->op_LeakyRelu;
			break;
		case 0x7c88793c: /* "Less" */
			rop = r->op_Less;
			break;
		case 0x0b8804e7: /* "Log" */
			rop = r->op_Log;
			break;
		case 0x7c88a33f: /* "Loop" */
			rop = r->op_Loop;
			break;
		case 0x07f77ce8: /* "LpNormalization" */
			rop = r->op_LpNormalization;
			break;
		case 0xc13f923b: /* "LpPool" */
			rop = r->op_LpPool;
			break;
		case 0xc2987915: /* "MatMul" */
			rop = r->op_MatMul;
			break;
		case 0x62fbd803: /* "MatMulInteger" */
			rop = r->op_MatMulInteger;
			break;
		case 0x0b88076b: /* "Max" */
			rop = r->op_Max;
			break;
		case 0x15f18a25: /* "MaxPool" */
			rop = r->op_MaxPool;
			break;
		case 0x018c06cf: /* "MaxRoiPool" */
			rop = r->op_MaxRoiPool;
			break;
		case 0x641501e8: /* "MaxUnpool" */
			rop = r->op_MaxUnpool;
			break;
		case 0x7c890346: /* "Mean" */
			rop = r->op_Mean;
			break;
		case 0x0b880869: /* "Min" */
			rop = r->op_Min;
			break;
		case 0x0b880925: /* "Mod" */
			rop = r->op_Mod;
			break;
		case 0x0b8809f3: /* "Mul" */
			rop = r->op_Mul;
			break;
		case 0xaec55410: /* "Multinomial" */
			rop = r->op_Multinomial;
			break;
		case 0x0b880c1f: /* "Neg" */
			rop = r->op_Neg;
			break;
		case 0x254e25a1: /* "NonMaxSuppression" */
			rop = r->op_NonMaxSuppression;
			break;
		case 0x82e45c50: /* "NonZero" */
			rop = r->op_NonZero;
			break;
		case 0x0b880d76: /* "Not" */
			rop = r->op_Not;
			break;
		case 0xc825b932: /* "OneHot" */
			rop = r->op_OneHot;
			break;
		case 0x005974e6: /* "Or" */
			rop = r->op_Or;
			break;
		case 0x0dd55b8d: /* "PRelu" */
			rop = r->op_PRelu;
			break;
		case 0x0b88141a: /* "Pad" */
			rop = r->op_Pad;
			break;
		case 0x0b8815fb: /* "Pow" */
			rop = r->op_Pow;
			break;
		case 0xe569f427: /* "QLinearConv" */
			rop = r->op_QLinearConv;
			break;
		case 0xfe108481: /* "QLinearMatMul" */
			rop = r->op_QLinearMatMul;
			break;
		case 0x37138211: /* "QuantizeLinear" */
			rop = r->op_QuantizeLinear;
			break;
		case 0x0b881a13: /* "RNN" */
			rop = r->op_RNN;
			break;
		case 0xc100684f: /* "RandomNormal" */
			rop = r->op_RandomNormal;
			break;
		case 0xa0b57174: /* "RandomNormalLike" */
			rop = r->op_RandomNormalLike;
			break;
		case 0xf8e97c66: /* "RandomUniform" */
			rop = r->op_RandomUniform;
			break;
		case 0x10a8b90b: /* "RandomUniformLike" */
			rop = r->op_RandomUniformLike;
			break;
		case 0x73d06f69: /* "Reciprocal" */
			rop = r->op_Reciprocal;
			break;
		case 0x7944853a: /* "ReduceL1" */
			rop = r->op_ReduceL1;
			break;
		case 0x7944853b: /* "ReduceL2" */
			rop = r->op_ReduceL2;
			break;
		case 0xeab46d14: /* "ReduceLogSum" */
			rop = r->op_ReduceLogSum;
			break;
		case 0x9a057a01: /* "ReduceLogSumExp" */
			rop = r->op_ReduceLogSumExp;
			break;
		case 0xa1d53763: /* "ReduceMax" */
			rop = r->op_ReduceMax;
			break;
		case 0xdc7c323e: /* "ReduceMean" */
			rop = r->op_ReduceMean;
			break;
		case 0xa1d53861: /* "ReduceMin" */
			rop = r->op_ReduceMin;
			break;
		case 0xdc7e1072: /* "ReduceProd" */
			rop = r->op_ReduceProd;
			break;
		case 0xa1d55372: /* "ReduceSum" */
			rop = r->op_ReduceSum;
			break;
		case 0x20917223: /* "ReduceSumSquare" */
			rop = r->op_ReduceSumSquare;
			break;
		case 0x7c8bc29d: /* "Relu" */
			rop = r->op_Relu;
			break;
		case 0x9fdbcf8d: /* "Reshape" */
			rop = r->op_Reshape;
			break;
		case 0xce8a9197: /* "Resize" */
			rop = r->op_Resize;
			break;
		case 0x5d77301a: /* "ReverseSequence" */
			rop = r->op_ReverseSequence;
			break;
		case 0x830cb9da: /* "RoiAlign" */
			rop = r->op_RoiAlign;
			break;
		case 0x0e09b7cd: /* "Round" */
			rop = r->op_Round;
			break;
		case 0x7c8c450a: /* "Scan" */
			rop = r->op_Scan;
			break;
		case 0xb4db6f18: /* "ScatterElements" */
			rop = r->op_ScatterElements;
			break;
		case 0x55be5b0d: /* "ScatterND" */
			rop = r->op_ScatterND;
			break;
		case 0x7c8c4efe: /* "Selu" */
			rop = r->op_Selu;
			break;
		case 0xe537ccd3: /* "SequenceAt" */
			rop = r->op_SequenceAt;
			break;
		case 0xa52772e3: /* "SequenceConstruct" */
			rop = r->op_SequenceConstruct;
			break;
		case 0x5e6e772d: /* "SequenceEmpty" */
			rop = r->op_SequenceEmpty;
			break;
		case 0x5e70f50e: /* "SequenceErase" */
			rop = r->op_SequenceErase;
			break;
		case 0x35a57cb3: /* "SequenceInsert" */
			rop = r->op_SequenceInsert;
			break;
		case 0x3bff64e0: /* "SequenceLength" */
			rop = r->op_SequenceLength;
			break;
		case 0x0e17a4d6: /* "Shape" */
			rop = r->op_Shape;
			break;
		case 0xd11575d4: /* "Shrink" */
			rop = r->op_Shrink;
			break;
		case 0xf5548151: /* "Sigmoid" */
			rop = r->op_Sigmoid;
			break;
		case 0x7c8c5f56: /* "Sign" */
			rop = r->op_Sign;
			break;
		case 0x0b8821ef: /* "Sin" */
			rop = r->op_Sin;
			break;
		case 0x7c8c6037: /* "Sinh" */
			rop = r->op_Sinh;
			break;
		case 0x7c8c61c0: /* "Size" */
			rop = r->op_Size;
			break;
		case 0x0e19f6b5: /* "Slice" */
			rop = r->op_Slice;
			break;
		case 0x6bec36a5: /* "Softplus" */
			rop = r->op_Softplus;
			break;
		case 0x6bedcd32: /* "Softsign" */
			rop = r->op_Softsign;
			break;
		case 0xa4436289: /* "SpaceToDepth" */
			rop = r->op_SpaceToDepth;
			break;
		case 0x0e1c35d1: /* "Split" */
			rop = r->op_Split;
			break;
		case 0x50e66fcd: /* "SplitToSequence" */
			rop = r->op_SplitToSequence;
			break;
		case 0x7c8c82cf: /* "Sqrt" */
			rop = r->op_Sqrt;
			break;
		case 0x08f69207: /* "Squeeze" */
			rop = r->op_Squeeze;
			break;
		case 0xf404645f: /* "StringNormalizer" */
			rop = r->op_StringNormalizer;
			break;
		case 0x0b88236f: /* "Sub" */
			rop = r->op_Sub;
			break;
		case 0x0b88237a: /* "Sum" */
			rop = r->op_Sum;
			break;
		case 0x0b882528: /* "Tan" */
			rop = r->op_Tan;
			break;
		case 0x7c8cca90: /* "Tanh" */
			rop = r->op_Tanh;
			break;
		case 0x46fbf3df: /* "TfIdfVectorizer" */
			rop = r->op_TfIdfVectorizer;
			break;
		case 0xa646ea33: /* "ThresholdedRelu" */
			rop = r->op_ThresholdedRelu;
			break;
		case 0x7c8cec53: /* "Tile" */
			rop = r->op_Tile;
			break;
		case 0x7c8d0643: /* "TopK" */
			rop = r->op_TopK;
			break;
		case 0x940b3944: /* "Transpose" */
			rop = r->op_Transpose;
			break;
		case 0xd6278d9c: /* "Unique" */
			rop = r->op_Unique;
			break;
		case 0xc836156a: /* "Unsqueeze" */
			rop = r->op_Unsqueeze;
			break;
		case 0x0e601820: /* "Where" */
			rop = r->op_Where;
			break;
		case 0x0b8837fe: /* "Xor" */
			rop = r->op_Xor;
			break;

		case 0x7c8388ee: /* "Celu" */
			rop = r->op_Celu;
			break;
		case 0x718dbc56: /* "DynamicQuantizeLinear" */
			rop = r->op_DynamicQuantizeLinear;
			break;
		case 0x7b2541c8: /* "GreaterOrEqual" */
			rop = r->op_GreaterOrEqual;
			break;
		case 0x60d9a535: /* "LessOrEqual" */
			rop = r->op_LessOrEqual;
			break;
		case 0xf8c82769: /* "LogSoftmax" */
			rop = r->op_LogSoftmax;
			break;
		case 0xbb8f2396: /* "MeanVarianceNormalization" */
			rop = r->op_MeanVarianceNormalization;
			break;
		case 0x6ed111df: /* "NegativeLogLikelihoodLoss" */
			rop = r->op_NegativeLogLikelihoodLoss;
			break;
		case 0x0e01ebd2: /* "Range" */
			rop = r->op_Range;
			break;
		case 0x034529c7: /* "Softmax" */
			rop = r->op_Softmax;
			break;
		case 0x522154a3: /* "SoftmaxCrossEntropyLoss" */
			rop = r->op_SoftmaxCrossEntropyLoss;
			break;

		default:
			rop = NULL;
			break;
		}
		if(rop)
			rop(n);
	}
}

static struct onnx_tensor_t * onnx_tensor_alloc_from_value_info(Onnx__ValueInfoProto * v)
{
	struct onnx_tensor_t * t;
	enum onnx_tensor_type_t type;
	int64_t * dims = NULL;
	int ndim;
	int i;

	if(!v || !v->name)
		return NULL;

	switch(v->type->value_case)
	{
	case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
		type = (enum onnx_tensor_type_t)v->type->tensor_type->elem_type;
		ndim = v->type->tensor_type->shape->n_dim;
		if(ndim > 0)
		{
			dims = memalign(8, sizeof(int64_t) * ndim);
			if(dims)
			{
				for(i = 0; i < ndim; i++)
				{
					switch(v->type->tensor_type->shape->dim[i]->value_case)
					{
					case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
						dims[i] = v->type->tensor_type->shape->dim[i]->dim_value;
						break;
					case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
						break;
					default:
						dims[i] = 0;
						break;
					}
				}
			}
		}
		t = onnx_tensor_alloc(v->name, type, dims, ndim);
		if(dims)
			free(dims);
		break;
	case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
		t = NULL;
		break;
	case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
		t = NULL;
		break;
	default:
		t = NULL;
		break;
	}
	return t;
}

static void onnx_tensor_copy_from_tensor_proto(struct onnx_tensor_t * t, Onnx__TensorProto * o)
{
	int sz, n;
	int i;

	if(t && o)
	{
		if(t->type == o->data_type)
		{
			sz = onnx_tensor_type_tosize(t->type);
			if(sz > 0)
			{
				if((o->raw_data.len > 0) && o->raw_data.data)
				{
					switch(o->data_type)
					{
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
						{
							float * p = (float *)t->datas;
							uint32_t * q = (uint32_t *)o->raw_data.data;
							union { uint32_t u; float f; } v;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
								{
									v.u = le32_to_cpu(q[i]);
									p[i] = v.f;
								}
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								v.u = le32_to_cpu(q[0]);
								t->scalar.v_float32 = v.f;
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
						{
							uint8_t * p = (uint8_t *)t->datas;
							uint8_t * q = (uint8_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len);
								memcpy(p, q, n);
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								t->scalar.v_uint8 = q[0];
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
						{
							int8_t * p = (int8_t *)t->datas;
							int8_t * q = (int8_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len);
								memcpy(p, q, n);
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								t->scalar.v_int8 = q[0];
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
						{
							uint16_t * p = (uint16_t *)t->datas;
							uint16_t * q = (uint16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								t->scalar.v_uint16 = le16_to_cpu(q[0]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
						{
							int16_t * p = (int16_t *)t->datas;
							int16_t * q = (int16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								t->scalar.v_int16 = le16_to_cpu(q[0]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
						{
							int32_t * p = (int32_t *)t->datas;
							int32_t * q = (int32_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le32_to_cpu(q[i]);
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								t->scalar.v_int32 = le32_to_cpu(q[0]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
						{
							int64_t * p = (int64_t *)t->datas;
							int64_t * q = (int64_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le64_to_cpu(q[i]);
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								t->scalar.v_int64 = le64_to_cpu(q[0]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
						{
							uint8_t * p = (uint8_t *)t->datas;
							uint8_t * q = (uint8_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len);
								memcpy(p, q, n);
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								t->scalar.v_bool = q[0] ? 1 : 0;
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
						{
							uint16_t * p = (uint16_t *)t->datas;
							uint16_t * q = (uint16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								t->scalar.v_float16 = le16_to_cpu(q[0]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
						{
							double * p = (double *)t->datas;
							uint64_t * q = (uint64_t *)o->raw_data.data;
							union { uint64_t u; double f; } v;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
								{
									v.u = le64_to_cpu(q[i]);
									p[i] = v.f;
								}
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								v.u = le64_to_cpu(q[0]);
								t->scalar.v_float64 = v.f;
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
						{
							uint32_t * p = (uint32_t *)t->datas;
							uint32_t * q = (uint32_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le32_to_cpu(q[i]);
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								t->scalar.v_uint32 = le32_to_cpu(q[0]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
						{
							uint64_t * p = (uint64_t *)t->datas;
							uint64_t * q = (uint64_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le64_to_cpu(q[i]);
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								t->scalar.v_uint64 = le64_to_cpu(q[0]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
						{
							float * p = (float *)t->datas;
							uint32_t * q = (uint32_t *)o->raw_data.data;
							union { uint32_t u; float f; } v;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz) * 2;
								for(i = 0; i < n; i++)
								{
									v.u = le32_to_cpu(q[i]);
									p[i] = v.f;
								}
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								v.u = le32_to_cpu(q[0]);
								t->scalar.v_complex64.real = v.f;
								v.u = le32_to_cpu(q[1]);
								t->scalar.v_complex64.imaginary = v.f;
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
						{
							double * p = (double *)t->datas;
							uint64_t * q = (uint64_t *)o->raw_data.data;
							union { uint64_t u; double f; } v;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz) * 2;
								for(i = 0; i < n; i++)
								{
									v.u = le64_to_cpu(q[i]);
									p[i] = v.f;
								}
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								v.u = le64_to_cpu(q[0]);
								t->scalar.v_complex64.real = v.f;
								v.u = le64_to_cpu(q[1]);
								t->scalar.v_complex64.imaginary = v.f;
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
						{
							uint16_t * p = (uint16_t *)t->datas;
							uint16_t * q = (uint16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (int)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
							}
							else if((t->ndata == 0) && ((int)o->raw_data.len == sz))
							{
								t->scalar.v_bfloat16 = le16_to_cpu(q[0]);
							}
						}
						break;
					default:
						break;
					}
				}
				else
				{
					switch(o->data_type)
					{
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
						n = min(t->ndata, (int)o->n_float_data);
						if((n > 0) && t->datas && o->float_data)
							memcpy(t->datas, o->float_data, sz * n);
						break;

					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
					case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
					case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
						//TODO
						n = min(t->ndata, (int)o->n_int32_data);
						if((n > 0) && t->datas && o->int32_data)
							memcpy(t->datas, o->int32_data, sz * n);
						break;

					case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
						n = min(t->ndata, (int)o->n_string_data);
						if((n > 0) && t->datas && o->string_data)
						{
							char ** str = (char **)t->datas;
							for(i = 0; i < t->ndata; i++)
							{
								if(str[i])
								{
									free(str[i]);
									str[i] = NULL;
								}
							}
							for(i = 0; i < n; i++)
							{
								str[i] = malloc(o->string_data[i].len + 1);
								if(str[i])
								{
									str[i][o->string_data[i].len] = 0;
									memcpy(str[i], o->string_data[i].data, o->string_data[i].len);
								}
							}
						}
						break;

					case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
						n = min(t->ndata, (int)o->n_int64_data);
						if((n > 0) && t->datas && o->int64_data)
							memcpy(t->datas, o->int64_data, sz * n);
						break;

					case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
						n = min(t->ndata, (int)o->n_double_data);
						if((n > 0) && t->datas && o->double_data)
							memcpy(t->datas, o->double_data, sz * n);
						break;

					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
						//TODO
						n = min(t->ndata, (int)o->n_uint64_data);
						if((n > 0) && t->datas && o->uint64_data)
							memcpy(t->datas, o->uint64_data, sz * n);
						break;

					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
						n = min(t->ndata, (int)(o->n_float_data / 2));
						if((n > 0) && t->datas && o->float_data)
							memcpy(t->datas, o->float_data, sz * n);
						break;

					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
						n = min(t->ndata, (int)(o->n_double_data / 2));
						if((n > 0) && t->datas && o->double_data)
							memcpy(t->datas, o->double_data, sz * n);
						break;

					default:
						break;
					}
				}
			}
		}
	}
}

static void hmap_entry_callback(struct hmap_entry_t * e)
{
	if(e && e->value)
		onnx_tensor_free((struct onnx_tensor_t *)e->value);
}

static void op_dummy(struct onnx_node_t * n)
{
}

struct onnx_context_t * onnx_context_alloc(const void * buf, size_t len, struct resolver_t * r)
{
	struct onnx_context_t * ctx;
	struct onnx_node_t * n;
	struct onnx_tensor_t * t;
	Onnx__ValueInfoProto * v;
	char * name;
	int i, j;

	if(!buf || len <= 0)
		return NULL;

	ctx = malloc(sizeof(struct onnx_context_t));
	if(!ctx)
		return NULL;

	ctx->model = onnx__model_proto__unpack(NULL, len, buf);
	if(!ctx->model)
	{
		free(ctx);
		return NULL;
	}

	ctx->nlen = ctx->model->graph->n_node;
	ctx->nodes = malloc(sizeof(struct onnx_node_t) * ctx->nlen);
	if(!ctx->nodes)
	{
		onnx__model_proto__free_unpacked(ctx->model, NULL);
		free(ctx);
		return NULL;
	}

	ctx->map = hmap_alloc(0);
	if(!ctx->map)
	{
		free(ctx->nodes);
		onnx__model_proto__free_unpacked(ctx->model, NULL);
		free(ctx);
		return NULL;
	}

	for(i = 0; i < ctx->model->graph->n_input; i++)
	{
		v = ctx->model->graph->input[i];
		if(!onnx_search_tensor(ctx, v->name))
		{
			t = onnx_tensor_alloc_from_value_info(v);
			if(t)
			{
				for(j = 0; j < ctx->model->graph->n_initializer; j++)
				{
					if(strcmp(ctx->model->graph->initializer[j]->name, t->name) == 0)
						onnx_tensor_copy_from_tensor_proto(t, ctx->model->graph->initializer[j]);
				}
				hmap_add(ctx->map, t->name, t);
			}
		}
	}

	for(i = 0; i < ctx->model->graph->n_output; i++)
	{
		v = ctx->model->graph->output[i];
		if(!onnx_search_tensor(ctx, v->name))
		{
			t = onnx_tensor_alloc_from_value_info(v);
			if(t)
				hmap_add(ctx->map, t->name, t);
		}
	}

	for(i = 0; i < ctx->model->graph->n_value_info; i++)
	{
		v = ctx->model->graph->value_info[i];
		if(!onnx_search_tensor(ctx, v->name))
		{
			t = onnx_tensor_alloc_from_value_info(v);
			if(t)
				hmap_add(ctx->map, t->name, t);
		}
	}

	for(i = 0; i < ctx->model->graph->n_node; i++)
	{
		for(j = 0; j < ctx->model->graph->node[i]->n_output; j++)
		{
			name = ctx->model->graph->node[i]->output[j];
			if(!onnx_search_tensor(ctx, name))
			{
				t = onnx_tensor_alloc(name, ONNX_TENSOR_TYPE_UNDEFINED, NULL, 0);
				if(t)
					hmap_add(ctx->map, name, t);
			}
		}
	}

	for(i = 0; i < ctx->model->graph->n_node; i++)
	{
		for(j = 0; j < ctx->model->graph->node[i]->n_input; j++)
		{
			name = ctx->model->graph->node[i]->input[j];
			if(!onnx_search_tensor(ctx, name))
			{
				hmap_free(ctx->map, hmap_entry_callback);
				free(ctx->nodes);
				onnx__model_proto__free_unpacked(ctx->model, NULL);
				free(ctx);
				return NULL;
			}
		}
	}

	for(i = 0; i < ctx->nlen; i++)
	{
		n = &ctx->nodes[i];
		memset(n, 0, sizeof(struct onnx_node_t));

		n->ctx = ctx;
		n->proto = ctx->model->graph->node[i];
		n->inputs = malloc(sizeof(Onnx__TensorProto *) * n->proto->n_input);
		if(n->inputs)
		{
			n->ninput = n->proto->n_input;
			for(j = 0; j < n->ninput; j++)
				n->inputs[j] = onnx_search_tensor(ctx, n->proto->input[j]);
		}
		n->outputs = malloc(sizeof(Onnx__TensorProto *) * n->proto->n_output);
		if(n->outputs)
		{
			n->noutput = n->proto->n_output;
			for(j = 0; j < n->noutput; j++)
				n->outputs[j] = onnx_search_tensor(ctx, n->proto->output[j]);
		}
		if(r)
			resolver_solve_operator(r, n);
		if(!n->op)
			resolver_solve_operator(&default_resolver, n);
		if(!n->op)
			n->op = op_dummy;
		if(n->init)
			n->init(n);
		if(n->op)
			n->op(n);
	}

	return ctx;
}

struct onnx_context_t * onnx_context_alloc_from_file(const char * filename, struct resolver_t * r)
{
	struct onnx_context_t * ctx = NULL;
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
				ctx = onnx_context_alloc(buf, len, r);
				free(buf);
			}
		}
		fclose(fp);
	}
	return ctx;
}

void onnx_context_free(struct onnx_context_t * ctx)
{
	struct onnx_node_t * n;
	int i;

	if(ctx)
	{
		if(ctx->map)
			hmap_free(ctx->map, hmap_entry_callback);
		if(ctx->nodes)
		{
			for(i = 0; i < ctx->nlen; i++)
			{
				n = &ctx->nodes[i];
				if(n->exit)
					n->exit(n);
				if(n->inputs)
					free(n->inputs);
				if(n->outputs)
					free(n->outputs);
			}
			free(ctx->nodes);
		}
		if(ctx->model)
			onnx__model_proto__free_unpacked(ctx->model, NULL);
		free(ctx);
	}
}

const char * onnx_tensor_type_tostring(enum onnx_tensor_type_t type)
{
	static const char * typestr[17] = {
		"undefined",
		"float32",
		"uint8",
		"int8",
		"uint16",
		"int16",
		"int32",
		"int64",
		"string",
		"bool",
		"float16",
		"float64",
		"uint32",
		"uint64",
		"complex64",
		"complex128",
		"bfloat16",
	};
	if((type > 0) && (type < (sizeof(typestr) / sizeof((typestr)[0]))))
		return typestr[type];
	return typestr[0];
}

int onnx_tensor_type_tosize(enum onnx_tensor_type_t type)
{
	static const int typesz[17] = {
		0,
		sizeof(float),
		sizeof(uint8_t),
		sizeof(int8_t),
		sizeof(uint16_t),
		sizeof(int16_t),
		sizeof(int32_t),
		sizeof(int64_t),
		sizeof(char *),
		sizeof(uint8_t),
		sizeof(uint16_t),
		sizeof(double),
		sizeof(uint32_t),
		sizeof(uint64_t),
		sizeof(float) * 2,
		sizeof(double) * 2,
		sizeof(uint16_t),
	};
	if((type > 0) && (type < (sizeof(typesz) / sizeof((typesz)[0]))))
		return typesz[type];
	return typesz[0];
}

struct onnx_tensor_t * onnx_search_tensor(struct onnx_context_t * ctx, const char * name)
{
	if(ctx)
		return hmap_search(ctx->map, name);
	return NULL;
}

struct onnx_tensor_t * onnx_tensor_alloc(const char * name, enum onnx_tensor_type_t type, int64_t * dims, int ndim)
{
	struct onnx_tensor_t * t;

	if(!name)
		return NULL;

	t = memalign(8, sizeof(struct onnx_tensor_t));
	if(!t)
		return NULL;
	memset(t, 0, sizeof(struct onnx_tensor_t));

	t->name = strdup(name);
	onnx_tensor_reinit(t, type, dims, ndim);
	return t;
}

struct onnx_tensor_t * onnx_tensor_alloc_from_file(const char * filename)
{
	struct onnx_tensor_t * t = NULL;
	Onnx__TensorProto * pb;
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
				pb = onnx__tensor_proto__unpack(NULL, len, buf);
				free(buf);
				if(pb)
				{
					t = onnx_tensor_alloc(pb->name, (enum onnx_tensor_type_t)pb->data_type, pb->dims, pb->n_dims);
					onnx_tensor_copy_from_tensor_proto(t, pb);
					onnx__tensor_proto__free_unpacked(pb, NULL);
				}
			}
		}
		fclose(fp);
	}
	return t;
}

void onnx_tensor_free(struct onnx_tensor_t * t)
{
	char ** str;
	int i;

	if(t)
	{
		if(t->name)
			free(t->name);
		if((t->ndim > 0) && t->dims)
			free(t->dims);
		if((t->ndata > 0) && t->datas)
		{
			if(t->type == ONNX_TENSOR_TYPE_STRING)
			{
				str = (char **)t->datas;
				for(i = 0; i < t->ndata; i++)
				{
					if(str[i])
						free(str[i]);
				}
			}
			free(t->datas);
		}
		free(t);
	}
}

void onnx_tensor_reinit(struct onnx_tensor_t * t, enum onnx_tensor_type_t type, int64_t * dims, int ndim)
{
	char ** str;
	int n, sz;
	int i;

	if(t)
	{
		if((t->ndim > 0) && t->dims)
		{
			free(t->dims);
			t->dims = NULL;
			t->ndim = 0;
		}
		if((t->ndata > 0) && t->datas)
		{
			if(t->type == ONNX_TENSOR_TYPE_STRING)
			{
				str = (char **)t->datas;
				for(i = 0; i < t->ndata; i++)
				{
					if(str[i])
					{
						free(str[i]);
						str[i] = NULL;
					}
				}
			}
			free(t->datas);
			t->datas = NULL;
			t->ndata = 0;
		}
		t->type = type;
		if(t->type != ONNX_TENSOR_TYPE_UNDEFINED)
		{
			if((ndim > 0) && dims)
			{
				t->dims = memalign(8, sizeof(int64_t) * ndim);
				if(t->dims)
				{
					memcpy(t->dims, dims, sizeof(int64_t) * ndim);
					t->ndim = ndim;
					for(i = 0, n = 1; i < t->ndim; i++)
					{
						if(t->dims[i] != 0)
							n *= t->dims[i];
					}
					sz = onnx_tensor_type_tosize(t->type);
					if(sz > 0)
					{
						t->datas = memalign(8, n * sz);
						if(t->datas)
						{
							memset(t->datas, 0, n * sz);
							t->ndata = n;
						}
					}
				}
			}
		}
	}
}

void onnx_tensor_apply(struct onnx_tensor_t * t, void * buf, int len, union onnx_scalar_t * s)
{
	int sz, l;
	int i;

	if(t)
	{
		if(t->datas && buf && (len > 0))
		{
			sz = onnx_tensor_type_tosize(t->type);
			if(sz > 0)
			{
				if(t->type == ONNX_TENSOR_TYPE_STRING)
				{
					char ** p = (char **)t->datas;
					char ** q = (char **)buf;
					for(i = 0; i < t->ndata; i++)
					{
						if(p[i])
						{
							free(p[i]);
							p[i] = NULL;
						}
					}
					l = min(t->ndata, len);
					for(i = 0; i < l; i++)
						p[i] = strdup(q[i]);
				}
				else
				{
					l = t->ndata * sz;
					if(l > 0)
						memcpy(t->datas, buf, min(l, len));
				}
			}
		}
		if(s)
			memcpy(&t->scalar, s, sizeof(union onnx_scalar_t));
	}
}

static Onnx__AttributeProto * onnx_search_attribute(struct onnx_node_t * n, const char * name)
{
	Onnx__AttributeProto * a;
	int i;

	if(n && name)
	{
		for(i = 0; i < n->proto->n_attribute; i++)
		{
			a = n->proto->attribute[i];
			if(strcmp(a->name, name) == 0)
				return a;
		}
	}
	return NULL;
}

float onnx_attribute_read_float(struct onnx_node_t * n, const char * name, float def)
{
	Onnx__AttributeProto * a = onnx_search_attribute(n, name);

	if(a && (a->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT))
		return a->f;
	return def;
}

int64_t onnx_attribute_read_int(struct onnx_node_t * n, const char * name, int64_t def)
{
	Onnx__AttributeProto * a = onnx_search_attribute(n, name);

	if(a && (a->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT))
		return a->i;
	return def;
}

char * onnx_attribute_read_string(struct onnx_node_t * n, const char * name, char * def)
{
	Onnx__AttributeProto * a = onnx_search_attribute(n, name);

	if(a && (a->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING))
	{
		if(a->s.len > 0)
			return (char *)a->s.data;
	}
	return def;
}

Onnx__TensorProto * onnx_attribute_read_tensor(struct onnx_node_t * n, const char * name, Onnx__TensorProto * def)
{
	Onnx__AttributeProto * a = onnx_search_attribute(n, name);

	if(a && (a->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR))
	{
		if(a->t)
			return a->t;
	}
	return def;
}

Onnx__GraphProto * onnx_attribute_read_graph(struct onnx_node_t * n, const char * name, Onnx__GraphProto * def)
{
	Onnx__AttributeProto * a = onnx_search_attribute(n, name);

	if(a && (a->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH))
	{
		if(a->g)
			return a->g;
	}
	return def;
}

Onnx__SparseTensorProto * onnx_attribute_read_sparse_tensor(struct onnx_node_t * n, const char * name, Onnx__SparseTensorProto * def)
{
	Onnx__AttributeProto * a = onnx_search_attribute(n, name);

	if(a && (a->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR))
	{
		if(a->sparse_tensor)
			return a->sparse_tensor;
	}
	return def;
}

void onnx_run(struct onnx_context_t * ctx)
{
	struct onnx_node_t * n;
	int i;

	if(ctx)
	{
		for(i = 0; i < ctx->nlen; i++)
		{
			n = &ctx->nodes[i];
			n->op(n);
		}
	}
}

void onnx_tensor_dump(struct onnx_tensor_t * t, int detail)
{
	char * p;
	int sz, i;

	if(t)
	{
		ONNX_LOG("%s: %s", t->name, onnx_tensor_type_tostring(t->type));
		if(t->ndim > 0)
		{
			ONNX_LOG("[");
			for(i = 0; i < t->ndim; i++)
			{
				ONNX_LOG("%ld", t->dims[i]);
				if(i != t->ndim - 1)
					ONNX_LOG(" x ");
			}
			ONNX_LOG("]");
			if(detail)
			{
				ONNX_LOG(" = \r\n");
				ONNX_LOG("[\r\n");
				p = (void *)t->datas;
				sz = onnx_tensor_type_tosize(t->type);
				for(i = 0; i < t->ndata; i++, p += sz)
				{
					switch(t->type)
					{
					case ONNX_TENSOR_TYPE_BOOL:
						ONNX_LOG("%s,", *((uint8_t *)p) ? "true" : "false");
						break;
					case ONNX_TENSOR_TYPE_INT8:
						ONNX_LOG("%d,", *((int8_t *)p));
						break;
					case ONNX_TENSOR_TYPE_INT16:
						ONNX_LOG("%d,", *((int16_t *)p));
						break;
					case ONNX_TENSOR_TYPE_INT32:
						ONNX_LOG("%d,", *((int32_t *)p));
						break;
					case ONNX_TENSOR_TYPE_INT64:
						ONNX_LOG("%ld,", *((int64_t *)p));
						break;
					case ONNX_TENSOR_TYPE_UINT8:
						ONNX_LOG("%u,", *((uint8_t *)p));
						break;
					case ONNX_TENSOR_TYPE_UINT16:
						ONNX_LOG("%u,", *((uint16_t *)p));
						break;
					case ONNX_TENSOR_TYPE_UINT32:
						ONNX_LOG("%u,", *((uint32_t *)p));
						break;
					case ONNX_TENSOR_TYPE_UINT64:
						ONNX_LOG("%lu,", *((uint64_t *)p));
						break;
					case ONNX_TENSOR_TYPE_BFLOAT16:
						ONNX_LOG("%g,", bfloat16_to_float32(*((uint16_t *)p)));
						break;
					case ONNX_TENSOR_TYPE_FLOAT16:
						ONNX_LOG("%g,", float16_to_float32(*((uint16_t *)p)));
						break;
					case ONNX_TENSOR_TYPE_FLOAT32:
						ONNX_LOG("%g,", *((float *)p));
						break;
					case ONNX_TENSOR_TYPE_FLOAT64:
						ONNX_LOG("%g,", *((double *)p));
						break;
					case ONNX_TENSOR_TYPE_COMPLEX64:
						ONNX_LOG("%g + %gi,", *((float *)p), *((float *)(p + sizeof(float))));
						break;
					case ONNX_TENSOR_TYPE_COMPLEX128:
						ONNX_LOG("%g + %gi,", *((double *)p), *((double *)(p + sizeof(double))));
						break;
					case ONNX_TENSOR_TYPE_STRING:
						ONNX_LOG("%s,", (char *)p);
						break;
					default:
						ONNX_LOG("?,");
						break;
					}
					ONNX_LOG("\r\n");
				}
				ONNX_LOG("]\r\n");
			}
			else
			{
				ONNX_LOG(" = ");
				ONNX_LOG("[...]");
				ONNX_LOG("\r\n");
			}
		}
		else
		{
			ONNX_LOG(" = ");
			switch(t->type)
			{
			case ONNX_TENSOR_TYPE_BOOL:
				ONNX_LOG("%s", t->scalar.v_bool ? "true" : "false");
				break;
			case ONNX_TENSOR_TYPE_INT8:
				ONNX_LOG("%d", t->scalar.v_int8);
				break;
			case ONNX_TENSOR_TYPE_INT16:
				ONNX_LOG("%d", t->scalar.v_int16);
				break;
			case ONNX_TENSOR_TYPE_INT32:
				ONNX_LOG("%d", t->scalar.v_int32);
				break;
			case ONNX_TENSOR_TYPE_INT64:
				ONNX_LOG("%ld", t->scalar.v_int64);
				break;
			case ONNX_TENSOR_TYPE_UINT8:
				ONNX_LOG("%u", t->scalar.v_uint8);
				break;
			case ONNX_TENSOR_TYPE_UINT16:
				ONNX_LOG("%u", t->scalar.v_uint16);
				break;
			case ONNX_TENSOR_TYPE_UINT32:
				ONNX_LOG("%u", t->scalar.v_uint32);
				break;
			case ONNX_TENSOR_TYPE_UINT64:
				ONNX_LOG("%lu", t->scalar.v_uint64);
				break;
			case ONNX_TENSOR_TYPE_BFLOAT16:
				ONNX_LOG("%g", bfloat16_to_float32(t->scalar.v_bfloat16));
				break;
			case ONNX_TENSOR_TYPE_FLOAT16:
				ONNX_LOG("%g", float16_to_float32(t->scalar.v_float16));
				break;
			case ONNX_TENSOR_TYPE_FLOAT32:
				ONNX_LOG("%g", t->scalar.v_float32);
				break;
			case ONNX_TENSOR_TYPE_FLOAT64:
				ONNX_LOG("%g", t->scalar.v_float64);
				break;
			case ONNX_TENSOR_TYPE_COMPLEX64:
				ONNX_LOG("%g + %gi", t->scalar.v_complex64.real, t->scalar.v_complex64.imaginary);
				break;
			case ONNX_TENSOR_TYPE_COMPLEX128:
				ONNX_LOG("%g + %gi", t->scalar.v_complex64.real, t->scalar.v_complex64.imaginary);
				break;
			default:
				ONNX_LOG("?");
				break;
			}
			ONNX_LOG("\r\n");
		}
	}
}
