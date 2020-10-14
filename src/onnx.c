#include <onnx.h>

static const char * data_type_tostring[] = {
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

static const char * attribute_type_tostring[] = {
	"undefined",
	"float32",
	"int64",
	"string",
	"tensor",
	"graph",
	"float32[]",
	"int64[]",
	"string[]",
	"tensor[]",
	"graph[]",
	"sparse tensor",
	"sparse tensor[]",
};

static void onnx_dump_tensor_type(Onnx__TensorProto * t)
{
	int i;

	if(t)
	{
		printf("%s[", data_type_tostring[t->data_type]);
		for(i = 0; i < t->n_dims; i++)
		{
			printf("%ld", t->dims[i]);
			if(i != t->n_dims - 1)
				printf(" x ");
		}
		printf("]");
	}
}

static void onnx_dump_attribute_type(Onnx__AttributeProto * a)
{
	if(a)
		printf("%s", attribute_type_tostring[a->type]);
}

static Onnx__TensorProto * onnx_get_initializer(struct onnx_context_t * ctx, const char * name)
{
	Onnx__TensorProto ** initializer = ctx->model->graph->initializer;
	int i;

	for(i = 0; i < ctx->model->graph->n_initializer; i++)
	{
		if(strcmp(initializer[i]->name, name) == 0)
			return initializer[i];
	}
	return NULL;
}

static void onnx_node_op_dummy(struct onnx_node_t * n)
{
	int i;

	printf("[%s] OPERATOR NOT IMPLEMENTED\r\n", n->proto->op_type);
	printf("\tInput:\r\n");
	for(i = 0; i < n->ninput; i++)
	{
		printf("\t\t%s - ", n->input[i]->name);
		onnx_dump_tensor_type(n->input[i]);
		printf("\r\n");
	}
	printf("\tOutput:\r\n");
	for(i = 0; i < n->noutput; i++)
	{
		printf("\t\t%s - ", n->output[i]->name);
		onnx_dump_tensor_type(n->output[i]);
		printf("\r\n");
	}
	if(n->proto->n_attribute > 0)
	{
		printf("\tAttribute:\r\n");
		for(i = 0; i < n->proto->n_attribute; i++)
		{
			printf("\t\t%s - ", n->proto->attribute[i]->name);
			onnx_dump_attribute_type(n->proto->attribute[i]);
			printf("\r\n");
		}
	}
}

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

static void resolver_operator(struct resolver_t * r, struct onnx_node_t * n)
{
	void (*op)(struct onnx_node_t *);

	if(r && n)
	{
		switch(shash(n->proto->op_type))
		{
		case 0x0b87d47b: /* "Abs" */
			op = r->op_Abs;
			break;
		case 0x7c82680b: /* "Acos" */
			op = r->op_Acos;
			break;
		case 0x0ccf69d3: /* "Acosh" */
			op = r->op_Acosh;
			break;
		case 0x0b87d4ae: /* "Add" */
			op = r->op_Add;
			break;
		case 0x0b87d5f8: /* "And" */
			op = r->op_And;
			break;
		case 0xa7c70ea5: /* "ArgMax" */
			op = r->op_ArgMax;
			break;
		case 0xa7c70fa3: /* "ArgMin" */
			op = r->op_ArgMin;
			break;
		case 0x7c82ab50: /* "Asin" */
			op = r->op_Asin;
			break;
		case 0x0cd815b8: /* "Asinh" */
			op = r->op_Asinh;
			break;
		case 0x7c82ae89: /* "Atan" */
			op = r->op_Atan;
			break;
		case 0x0cd88011: /* "Atanh" */
			op = r->op_Atanh;
			break;
		case 0xf1a1e23a: /* "AveragePool" */
			op = r->op_AveragePool;
			break;
		case 0x2d3b46ee: /* "BatchNormalization" */
			op = r->op_BatchNormalization;
			break;
		case 0x0bfe45a2: /* "BitShift" */
			op = r->op_BitShift;
			break;
		case 0x7c8378d0: /* "Cast" */
			op = r->op_Cast;
			break;
		case 0x7c838882: /* "Ceil" */
			op = r->op_Ceil;
			break;
		case 0x7c83a64d: /* "Clip" */
			op = r->op_Clip;
			break;
		case 0xb7db9db1: /* "Compress" */
			op = r->op_Compress;
			break;
		case 0xac3f4a9d: /* "Concat" */
			op = r->op_Concat;
			break;
		case 0x5053caca: /* "ConcatFromSequence" */
			op = r->op_ConcatFromSequence;
			break;
		case 0xba6816ef: /* "Constant" */
			op = r->op_Constant;
			break;
		case 0xe468a875: /* "ConstantOfShape" */
			op = r->op_ConstantOfShape;
			break;
		case 0x7c83b3bb: /* "Conv" */
			op = r->op_Conv;
			break;
		case 0x8371dbe9: /* "ConvInteger" */
			op = r->op_ConvInteger;
			break;
		case 0x3903c4ba: /* "ConvTranspose" */
			op = r->op_ConvTranspose;
			break;
		case 0x0b87deaa: /* "Cos" */
			op = r->op_Cos;
			break;
		case 0x7c83b452: /* "Cosh" */
			op = r->op_Cosh;
			break;
		case 0xacab0fbf: /* "CumSum" */
			op = r->op_CumSum;
			break;
		case 0xc9c1d669: /* "DepthToSpace" */
			op = r->op_DepthToSpace;
			break;
		case 0xf9cc985a: /* "DequantizeLinear" */
			op = r->op_DequantizeLinear;
			break;
		case 0x0b87e1a2: /* "Det" */
			op = r->op_Det;
			break;
		case 0x0b87e228: /* "Div" */
			op = r->op_Div;
			break;
		case 0x883bca72: /* "Dropout" */
			op = r->op_Dropout;
			break;
		case 0xb07d4f76: /* "Einsum" */
			op = r->op_Einsum;
			break;
		case 0x0b87e6cb: /* "Elu" */
			op = r->op_Elu;
			break;
		case 0x0d1f905d: /* "Equal" */
			op = r->op_Equal;
			break;
		case 0x0b87e782: /* "Erf" */
			op = r->op_Erf;
			break;
		case 0x0b87e852: /* "Exp" */
			op = r->op_Exp;
			break;
		case 0xb18d8a45: /* "Expand" */
			op = r->op_Expand;
			break;
		case 0xe4c1560d: /* "EyeLike" */
			op = r->op_EyeLike;
			break;
		case 0x13363dd3: /* "Flatten" */
			op = r->op_Flatten;
			break;
		case 0x0d2ed347: /* "Floor" */
			op = r->op_Floor;
			break;
		case 0x0b87ebd3: /* "GRU" */
			op = r->op_GRU;
			break;
		case 0xb499f620: /* "Gather" */
			op = r->op_Gather;
			break;
		case 0x7c94d43d: /* "GatherElements" */
			op = r->op_GatherElements;
			break;
		case 0x42f00872: /* "GatherND" */
			op = r->op_GatherND;
			break;
		case 0x7c85ba8b: /* "Gemm" */
			op = r->op_Gemm;
			break;
		case 0x9289c84b: /* "GlobalAveragePool" */
			op = r->op_GlobalAveragePool;
			break;
		case 0x3f5a29ac: /* "GlobalLpPool" */
			op = r->op_GlobalLpPool;
			break;
		case 0x575f0fb6: /* "GlobalMaxPool" */
			op = r->op_GlobalMaxPool;
			break;
		case 0x6e6d652f: /* "Greater" */
			op = r->op_Greater;
			break;
		case 0x10341df0: /* "HardSigmoid" */
			op = r->op_HardSigmoid;
			break;
		case 0x94acb4aa: /* "Hardmax" */
			op = r->op_Hardmax;
			break;
		case 0xdfd9b28f: /* "Identity" */
			op = r->op_Identity;
			break;
		case 0x00597414: /* "If" */
			op = r->op_If;
			break;
		case 0xfb0902c1: /* "InstanceNormalization" */
			op = r->op_InstanceNormalization;
			break;
		case 0x0d68519e: /* "IsInf" */
			op = r->op_IsInf;
			break;
		case 0x0d68651e: /* "IsNaN" */
			op = r->op_IsNaN;
			break;
		case 0x0b880111: /* "LRN" */
			op = r->op_LRN;
			break;
		case 0x7c882885: /* "LSTM" */
			op = r->op_LSTM;
			break;
		case 0xea2c5c33: /* "LeakyRelu" */
			op = r->op_LeakyRelu;
			break;
		case 0x7c88793c: /* "Less" */
			op = r->op_Less;
			break;
		case 0x0b8804e7: /* "Log" */
			op = r->op_Log;
			break;
		case 0x7c88a33f: /* "Loop" */
			op = r->op_Loop;
			break;
		case 0x07f77ce8: /* "LpNormalization" */
			op = r->op_LpNormalization;
			break;
		case 0xc13f923b: /* "LpPool" */
			op = r->op_LpPool;
			break;
		case 0xc2987915: /* "MatMul" */
			op = r->op_MatMul;
			break;
		case 0x62fbd803: /* "MatMulInteger" */
			op = r->op_MatMulInteger;
			break;
		case 0x0b88076b: /* "Max" */
			op = r->op_Max;
			break;
		case 0x15f18a25: /* "MaxPool" */
			op = r->op_MaxPool;
			break;
		case 0x018c06cf: /* "MaxRoiPool" */
			op = r->op_MaxRoiPool;
			break;
		case 0x641501e8: /* "MaxUnpool" */
			op = r->op_MaxUnpool;
			break;
		case 0x7c890346: /* "Mean" */
			op = r->op_Mean;
			break;
		case 0x0b880869: /* "Min" */
			op = r->op_Min;
			break;
		case 0x0b880925: /* "Mod" */
			op = r->op_Mod;
			break;
		case 0x0b8809f3: /* "Mul" */
			op = r->op_Mul;
			break;
		case 0xaec55410: /* "Multinomial" */
			op = r->op_Multinomial;
			break;
		case 0x0b880c1f: /* "Neg" */
			op = r->op_Neg;
			break;
		case 0x254e25a1: /* "NonMaxSuppression" */
			op = r->op_NonMaxSuppression;
			break;
		case 0x82e45c50: /* "NonZero" */
			op = r->op_NonZero;
			break;
		case 0x0b880d76: /* "Not" */
			op = r->op_Not;
			break;
		case 0xc825b932: /* "OneHot" */
			op = r->op_OneHot;
			break;
		case 0x005974e6: /* "Or" */
			op = r->op_Or;
			break;
		case 0x0dd55b8d: /* "PRelu" */
			op = r->op_PRelu;
			break;
		case 0x0b88141a: /* "Pad" */
			op = r->op_Pad;
			break;
		case 0x0b8815fb: /* "Pow" */
			op = r->op_Pow;
			break;
		case 0xe569f427: /* "QLinearConv" */
			op = r->op_QLinearConv;
			break;
		case 0xfe108481: /* "QLinearMatMul" */
			op = r->op_QLinearMatMul;
			break;
		case 0x37138211: /* "QuantizeLinear" */
			op = r->op_QuantizeLinear;
			break;
		case 0x0b881a13: /* "RNN" */
			op = r->op_RNN;
			break;
		case 0xc100684f: /* "RandomNormal" */
			op = r->op_RandomNormal;
			break;
		case 0xa0b57174: /* "RandomNormalLike" */
			op = r->op_RandomNormalLike;
			break;
		case 0xf8e97c66: /* "RandomUniform" */
			op = r->op_RandomUniform;
			break;
		case 0x10a8b90b: /* "RandomUniformLike" */
			op = r->op_RandomUniformLike;
			break;
		case 0x73d06f69: /* "Reciprocal" */
			op = r->op_Reciprocal;
			break;
		case 0x7944853a: /* "ReduceL1" */
			op = r->op_ReduceL1;
			break;
		case 0x7944853b: /* "ReduceL2" */
			op = r->op_ReduceL2;
			break;
		case 0xeab46d14: /* "ReduceLogSum" */
			op = r->op_ReduceLogSum;
			break;
		case 0x9a057a01: /* "ReduceLogSumExp" */
			op = r->op_ReduceLogSumExp;
			break;
		case 0xa1d53763: /* "ReduceMax" */
			op = r->op_ReduceMax;
			break;
		case 0xdc7c323e: /* "ReduceMean" */
			op = r->op_ReduceMean;
			break;
		case 0xa1d53861: /* "ReduceMin" */
			op = r->op_ReduceMin;
			break;
		case 0xdc7e1072: /* "ReduceProd" */
			op = r->op_ReduceProd;
			break;
		case 0xa1d55372: /* "ReduceSum" */
			op = r->op_ReduceSum;
			break;
		case 0x20917223: /* "ReduceSumSquare" */
			op = r->op_ReduceSumSquare;
			break;
		case 0x7c8bc29d: /* "Relu" */
			op = r->op_Relu;
			break;
		case 0x9fdbcf8d: /* "Reshape" */
			op = r->op_Reshape;
			break;
		case 0xce8a9197: /* "Resize" */
			op = r->op_Resize;
			break;
		case 0x5d77301a: /* "ReverseSequence" */
			op = r->op_ReverseSequence;
			break;
		case 0x830cb9da: /* "RoiAlign" */
			op = r->op_RoiAlign;
			break;
		case 0x0e09b7cd: /* "Round" */
			op = r->op_Round;
			break;
		case 0x7c8c450a: /* "Scan" */
			op = r->op_Scan;
			break;
		case 0xb4db6f18: /* "ScatterElements" */
			op = r->op_ScatterElements;
			break;
		case 0x55be5b0d: /* "ScatterND" */
			op = r->op_ScatterND;
			break;
		case 0x7c8c4efe: /* "Selu" */
			op = r->op_Selu;
			break;
		case 0xe537ccd3: /* "SequenceAt" */
			op = r->op_SequenceAt;
			break;
		case 0xa52772e3: /* "SequenceConstruct" */
			op = r->op_SequenceConstruct;
			break;
		case 0x5e6e772d: /* "SequenceEmpty" */
			op = r->op_SequenceEmpty;
			break;
		case 0x5e70f50e: /* "SequenceErase" */
			op = r->op_SequenceErase;
			break;
		case 0x35a57cb3: /* "SequenceInsert" */
			op = r->op_SequenceInsert;
			break;
		case 0x3bff64e0: /* "SequenceLength" */
			op = r->op_SequenceLength;
			break;
		case 0x0e17a4d6: /* "Shape" */
			op = r->op_Shape;
			break;
		case 0xd11575d4: /* "Shrink" */
			op = r->op_Shrink;
			break;
		case 0xf5548151: /* "Sigmoid" */
			op = r->op_Sigmoid;
			break;
		case 0x7c8c5f56: /* "Sign" */
			op = r->op_Sign;
			break;
		case 0x0b8821ef: /* "Sin" */
			op = r->op_Sin;
			break;
		case 0x7c8c6037: /* "Sinh" */
			op = r->op_Sinh;
			break;
		case 0x7c8c61c0: /* "Size" */
			op = r->op_Size;
			break;
		case 0x0e19f6b5: /* "Slice" */
			op = r->op_Slice;
			break;
		case 0x6bec36a5: /* "Softplus" */
			op = r->op_Softplus;
			break;
		case 0x6bedcd32: /* "Softsign" */
			op = r->op_Softsign;
			break;
		case 0xa4436289: /* "SpaceToDepth" */
			op = r->op_SpaceToDepth;
			break;
		case 0x0e1c35d1: /* "Split" */
			op = r->op_Split;
			break;
		case 0x50e66fcd: /* "SplitToSequence" */
			op = r->op_SplitToSequence;
			break;
		case 0x7c8c82cf: /* "Sqrt" */
			op = r->op_Sqrt;
			break;
		case 0x08f69207: /* "Squeeze" */
			op = r->op_Squeeze;
			break;
		case 0xf404645f: /* "StringNormalizer" */
			op = r->op_StringNormalizer;
			break;
		case 0x0b88236f: /* "Sub" */
			op = r->op_Sub;
			break;
		case 0x0b88237a: /* "Sum" */
			op = r->op_Sum;
			break;
		case 0x0b882528: /* "Tan" */
			op = r->op_Tan;
			break;
		case 0x7c8cca90: /* "Tanh" */
			op = r->op_Tanh;
			break;
		case 0x46fbf3df: /* "TfIdfVectorizer" */
			op = r->op_TfIdfVectorizer;
			break;
		case 0xa646ea33: /* "ThresholdedRelu" */
			op = r->op_ThresholdedRelu;
			break;
		case 0x7c8cec53: /* "Tile" */
			op = r->op_Tile;
			break;
		case 0x7c8d0643: /* "TopK" */
			op = r->op_TopK;
			break;
		case 0x940b3944: /* "Transpose" */
			op = r->op_Transpose;
			break;
		case 0xd6278d9c: /* "Unique" */
			op = r->op_Unique;
			break;
		case 0xc836156a: /* "Unsqueeze" */
			op = r->op_Unsqueeze;
			break;
		case 0x0e601820: /* "Where" */
			op = r->op_Where;
			break;
		case 0x0b8837fe: /* "Xor" */
			op = r->op_Xor;
			break;

		case 0x7c8388ee: /* "Celu" */
			op = r->op_Celu;
			break;
		case 0x718dbc56: /* "DynamicQuantizeLinear" */
			op = r->op_DynamicQuantizeLinear;
			break;
		case 0x7b2541c8: /* "GreaterOrEqual" */
			op = r->op_GreaterOrEqual;
			break;
		case 0x60d9a535: /* "LessOrEqual" */
			op = r->op_LessOrEqual;
			break;
		case 0xf8c82769: /* "LogSoftmax" */
			op = r->op_LogSoftmax;
			break;
		case 0xbb8f2396: /* "MeanVarianceNormalization" */
			op = r->op_MeanVarianceNormalization;
			break;
		case 0x6ed111df: /* "NegativeLogLikelihoodLoss" */
			op = r->op_NegativeLogLikelihoodLoss;
			break;
		case 0x0e01ebd2: /* "Range" */
			op = r->op_Range;
			break;
		case 0x034529c7: /* "Softmax" */
			op = r->op_Softmax;
			break;
		case 0x522154a3: /* "SoftmaxCrossEntropyLoss" */
			op = r->op_SoftmaxCrossEntropyLoss;
			break;

		default:
			op = NULL;
			break;
		}
		if(op)
			op(n);
	}
}

static void hmap_entry_callback(struct hmap_entry_t * e)
{
	Onnx__TensorProto * t;

	if(e && (t = e->value))
		onnx_tensor_free(t);
}

struct onnx_context_t * onnx_context_alloc(const void * buf, size_t len, struct resolver_t * r)
{
	struct onnx_context_t * ctx;
	struct onnx_node_t * n;
	Onnx__TensorProto * t, * initial;
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
			t = onnx_tensor_alloc(v);
			if(t)
			{
				initial = onnx_get_initializer(ctx, t->name);
				if(initial)
				{
					//TODO Copy from initializer.
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
			t = onnx_tensor_alloc(v);
			if(t)
				hmap_add(ctx->map, t->name, t);
		}
	}

	for(i = 0; i < ctx->model->graph->n_value_info; i++)
	{
		v = ctx->model->graph->value_info[i];
		if(!onnx_search_tensor(ctx, v->name))
		{
			t = onnx_tensor_alloc(v);
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
				t = malloc(sizeof(Onnx__TensorProto));
				if(t)
				{
					onnx__tensor_proto__init(t);
					t->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED;
					t->name = strdup(name);
					t->doc_string = NULL;
					hmap_add(ctx->map, name, t);
				}
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
		n->input = malloc(sizeof(Onnx__TensorProto *) * n->proto->n_input);
		if(n->input)
		{
			n->ninput = n->proto->n_input;
			for(j = 0; j < n->ninput; j++)
				n->input[j] = onnx_search_tensor(ctx, n->proto->input[j]);
		}
		n->output = malloc(sizeof(Onnx__TensorProto *) * n->proto->n_output);
		if(n->output)
		{
			n->noutput = n->proto->n_output;
			for(j = 0; j < n->noutput; j++)
				n->output[j] = onnx_search_tensor(ctx, n->proto->output[j]);
		}
		if(r)
		{
			resolver_operator(r, n);
			if(!n->op)
				resolver_operator(&default_resolver, n);
		}
		else
		{
			resolver_operator(&default_resolver, n);
		}
		if(!n->op)
			n->op = onnx_node_op_dummy;
		if(n->init)
			n->init(n);
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
	}
	fclose(fp);
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
				if(n->input)
					free(n->input);
				if(n->output)
					free(n->output);
			}
			free(ctx->nodes);
		}
		if(ctx->model)
			onnx__model_proto__free_unpacked(ctx->model, NULL);
		free(ctx);
	}
}

Onnx__TensorProto * onnx_tensor_alloc(Onnx__ValueInfoProto * v)
{
	Onnx__TensorProto * t;
	int n, i;

	if(!v)
		return NULL;

	t = malloc(sizeof(Onnx__TensorProto));
	if(!t)
		return NULL;

	onnx__tensor_proto__init(t);
	switch(v->type->value_case)
	{
	case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
		t->n_dims = v->type->tensor_type->shape->n_dim;
		t->dims = malloc(sizeof(int64_t) * t->n_dims);
		for(i = 0; i < t->n_dims; i++)
		{
			switch(v->type->tensor_type->shape->dim[i]->value_case)
			{
			case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
				t->dims[i] = v->type->tensor_type->shape->dim[i]->dim_value;
				break;
			case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
				break;
			default:
				break;
			}
		}
		t->data_type = v->type->tensor_type->elem_type;
		break;
	case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
		break;
	case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
		break;
	default:
		break;
	}
	if(t->n_dims > 0)
	{
		for(i = 0, n = 1; i < t->n_dims; i++)
			n *= t->dims[i];
		switch(t->data_type)
		{
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
			t->float_data = malloc(sizeof(float) * n);
			if(t->float_data)
			{
				memset(t->float_data, 0, sizeof(float) * n);
				t->n_float_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
			t->int32_data = malloc(sizeof(uint8_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(uint8_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
			t->int32_data = malloc(sizeof(int8_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int8_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
			t->int32_data = malloc(sizeof(uint16_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(uint16_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
			t->int32_data = malloc(sizeof(int16_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int16_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
			t->int32_data = malloc(sizeof(int32_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int32_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
			t->int64_data = malloc(sizeof(int64_t) * n);
			if(t->int64_data)
			{
				memset(t->int64_data, 0, sizeof(int64_t) * n);
				t->n_int64_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
			t->string_data = malloc(sizeof(ProtobufCBinaryData) * n);
			if(t->string_data)
			{
				for(i = 0; i < n; i++)
				{
					t->string_data[i].len = 0;
					t->string_data[i].data = NULL;
				}
				t->n_string_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
			t->int32_data = malloc(sizeof(int32_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int32_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
			t->int32_data = malloc(sizeof(int16_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int16_t) * n);
				t->n_int32_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
			t->double_data = malloc(sizeof(double) * n);
			if(t->double_data)
			{
				memset(t->double_data, 0, sizeof(double) * n);
				t->n_double_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
			t->uint64_data = malloc(sizeof(uint32_t) * n);
			if(t->int64_data)
			{
				memset(t->uint64_data, 0, sizeof(uint32_t) * n);
				t->n_uint64_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
			t->uint64_data = malloc(sizeof(uint64_t) * n);
			if(t->int64_data)
			{
				memset(t->uint64_data, 0, sizeof(uint64_t) * n);
				t->n_uint64_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
			t->float_data = malloc(sizeof(float) * 2 * n);
			if(t->float_data)
			{
				memset(t->float_data, 0, sizeof(float) *2 * n);
				t->n_float_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
			t->double_data = malloc(sizeof(double) * 2 * n);
			if(t->double_data)
			{
				memset(t->double_data, 0, sizeof(double) * 2 * n);
				t->n_double_data = n;
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
			t->int32_data = malloc(sizeof(int16_t) * n);
			if(t->int32_data)
			{
				memset(t->int32_data, 0, sizeof(int16_t) * n);
				t->n_int32_data = n;
			}
			break;
		default:
			break;
		}
	}
	t->name = v->name ? strdup(v->name) : NULL;
	t->doc_string = v->doc_string ? strdup(v->doc_string) : NULL;
	return t;
}

void onnx_tensor_free(Onnx__TensorProto * t)
{
	int i;

	if(t)
	{
		if((t->n_dims > 0) && t->dims)
			free(t->dims);
		if(t->segment)
			free(t->segment);
		if((t->n_float_data > 0) && t->float_data)
			free(t->float_data);
		if((t->n_int32_data > 0) && t->int32_data)
			free(t->int32_data);
		if((t->n_string_data > 0) && t->string_data)
		{
			for(i = 0; i < t->n_string_data; i++)
			{
				if(t->string_data[i].data)
					free(t->string_data[i].data);
			}
			free(t->string_data);
		}
		if((t->n_int64_data > 0) && t->int64_data)
			free(t->int64_data);
		if(t->name)
			free(t->name);
		if(t->doc_string)
			free(t->doc_string);
		if((t->raw_data.len > 0) && t->raw_data.data)
			free(t->raw_data.data);
		if((t->n_external_data > 0) && t->external_data)
		{
			for(i = 0; i < t->n_external_data; i++)
			{
				if(t->external_data[i]->key)
					free(t->external_data[i]->key);
				if(t->external_data[i]->value)
					free(t->external_data[i]->value);
			}
			free(t->external_data);
		}
		if((t->n_double_data > 0) && t->double_data)
			free(t->double_data);
		if((t->n_uint64_data > 0) && t->uint64_data)
			free(t->uint64_data);
		free(t);
	}
}

Onnx__TensorProto * onnx_search_tensor(struct onnx_context_t * ctx, const char * name)
{
	if(ctx)
		return hmap_search(ctx->map, name);
	return NULL;
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

static Onnx__ValueInfoProto * onnx_get_input_value_info(struct onnx_context_t * ctx, const char * name)
{
	Onnx__ValueInfoProto ** input = ctx->model->graph->input;
	int i;

	for(i = 0; i < ctx->model->graph->n_input; i++)
	{
		if(strcmp(input[i]->name, name) == 0)
			return input[i];
	}
	return NULL;
}

static Onnx__ValueInfoProto * onnx_get_output_value_info(struct onnx_context_t * ctx, const char * name)
{
	Onnx__ValueInfoProto ** output = ctx->model->graph->output;
	int i;

	for(i = 0; i < ctx->model->graph->n_output; i++)
	{
		if(strcmp(output[i]->name, name) == 0)
			return output[i];
	}
	return NULL;
}

static Onnx__ValueInfoProto * onnx_get_value_info(struct onnx_context_t * ctx, const char * name)
{
	Onnx__ValueInfoProto ** value_info = ctx->model->graph->value_info;
	int i;

	for(i = 0; i < ctx->model->graph->n_value_info; i++)
	{
		if(strcmp(value_info[i]->name, name) == 0)
			return value_info[i];
	}
	return NULL;
}


static void onnx_dump_value_info_type(Onnx__ValueInfoProto * v)
{
	int i;

	if(v)
	{
		switch(v->type->value_case)
		{
		case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
			printf("%s[", data_type_tostring[v->type->tensor_type->elem_type]);
			for(i = 0; i < v->type->tensor_type->shape->n_dim; i++)
			{
				switch(v->type->tensor_type->shape->dim[i]->value_case)
				{
				case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
					printf("%ld", v->type->tensor_type->shape->dim[i]->dim_value);
					break;
				case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
					printf("%s", v->type->tensor_type->shape->dim[i]->dim_param);
					break;
				default:
					printf("?");
					break;
				}
				if(i != v->type->tensor_type->shape->n_dim - 1)
				{
					printf(" x ");
				}
			}
			printf("]");
			break;
		case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
			break;
		case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
			break;
		default:
			break;
		}
	}
}

void onnx_dump_model(struct onnx_context_t * ctx)
{
	Onnx__ModelProto * model;
	Onnx__TensorProto * t;
	Onnx__ValueInfoProto * v;
	int i, j;

	if(ctx && (model = ctx->model))
	{
		printf("IR Version: v%ld\r\n", model->ir_version);
		printf("Producer: %s %s\r\n", model->producer_name, model->producer_version);
		printf("Domain: %s\r\n", model->domain);
		printf("Imports:\r\n");
		for(i = 0; i < model->n_opset_import; i++)
			printf("\t%s v%ld\r\n", (strlen(model->opset_import[i]->domain) > 0) ? model->opset_import[i]->domain : "ai.onnx", model->opset_import[i]->version);

		printf("\r\nInitializer:\r\n");
		for(i = 0; i < model->graph->n_initializer; i++)
		{
			printf("\t%s - ", model->graph->initializer[i]->name);
			onnx_dump_tensor_type(model->graph->initializer[i]);
			printf("\r\n");
		}

		printf("\r\nInput:\r\n");
		for(i = 0; i < model->graph->n_input; i++)
		{
			printf("\t%s - ", model->graph->input[i]->name);
			onnx_dump_value_info_type(model->graph->input[i]);
			printf("\r\n");
		}

		printf("\r\nOutput:\r\n");
		for(i = 0; i < model->graph->n_output; i++)
		{
			printf("\t%s - ", model->graph->output[i]->name);
			onnx_dump_value_info_type(model->graph->output[i]);
			printf("\r\n");
		}

		printf("\r\nValueInfo:\r\n");
		for(i = 0; i < model->graph->n_value_info; i++)
		{
			printf("\t%s - ", model->graph->value_info[i]->name);
			onnx_dump_value_info_type(model->graph->value_info[i]);
			printf("\r\n");
		}

		printf("\r\nGraph node:\r\n");
		for(i = 0; i < model->graph->n_node; i++)
		{
			printf("[%s] - [%s]\r\n", model->graph->node[i]->name, model->graph->node[i]->op_type);
			printf("\tInputs:\r\n");
			for(j = 0; j < model->graph->node[i]->n_input; j++)
			{
				printf("\t\t%s - ", model->graph->node[i]->input[j]);
				v = onnx_get_input_value_info(ctx, model->graph->node[i]->input[j]);
				if(!v)
					v = onnx_get_value_info(ctx, model->graph->node[i]->input[j]);
				if(!v)
				{
					t = onnx_get_initializer(ctx, model->graph->node[i]->input[j]);
					if(t)
					{
						printf("initializer ");
						onnx_dump_tensor_type(t);
					}
				}
				else
					onnx_dump_value_info_type(v);
				printf("\r\n");
			}
			printf("\tOutputs:\r\n");
			for(j = 0; j < model->graph->node[i]->n_output; j++)
			{
				printf("\t\t%s - ", model->graph->node[i]->output[j]);
				v = onnx_get_output_value_info(ctx, model->graph->node[i]->output[j]);
				if(!v)
					v = onnx_get_value_info(ctx, model->graph->node[i]->output[j]);
				onnx_dump_value_info_type(v);
				printf("\r\n");
			}
			printf("\tAttributes:\r\n");
			for(j = 0; j < model->graph->node[i]->n_attribute; j++)
			{
				printf("\t\t%s - ", model->graph->node[i]->attribute[j]->name);
				onnx_dump_attribute_type(model->graph->node[i]->attribute[j]);
				printf("\r\n");
			}
		}
	}
}
