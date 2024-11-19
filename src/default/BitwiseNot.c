#include <onnx.h>

static int BitwiseNot_init(struct onnx_node_t * n)
{
	if((n->ninput == 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int BitwiseNot_exit(struct onnx_node_t * n)
{
	return 1;
}

static int BitwiseNot_reshape(struct onnx_node_t * n)
{
	return 1;
}

static void BitwiseNot_int8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int8_t * py = (int8_t *)y->datas;
	int8_t * px = (int8_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = ~px[i];
}

static void BitwiseNot_int16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int16_t * py = (int16_t *)y->datas;
	int16_t * px = (int16_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = ~px[i];
}

static void BitwiseNot_int32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int32_t * py = (int32_t *)y->datas;
	int32_t * px = (int32_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = ~px[i];
}

static void BitwiseNot_int64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	int64_t * py = (int64_t *)y->datas;
	int64_t * px = (int64_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = ~px[i];
}

static void BitwiseNot_uint8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * px = (uint8_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = ~px[i];
}

static void BitwiseNot_uint16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px = (uint16_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = ~px[i];
}

static void BitwiseNot_uint32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t * px = (uint32_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = ~px[i];
}

static void BitwiseNot_uint64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * px = (uint64_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = ~px[i];
}

void resolver_default_op_BitwiseNot(struct onnx_node_t * n)
{
	if(n->opset >= 18)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = BitwiseNot_init;
			n->exit = BitwiseNot_exit;
			n->reshape = BitwiseNot_reshape;
			n->operator = BitwiseNot_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = BitwiseNot_init;
			n->exit = BitwiseNot_exit;
			n->reshape = BitwiseNot_reshape;
			n->operator = BitwiseNot_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = BitwiseNot_init;
			n->exit = BitwiseNot_exit;
			n->reshape = BitwiseNot_reshape;
			n->operator = BitwiseNot_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = BitwiseNot_init;
			n->exit = BitwiseNot_exit;
			n->reshape = BitwiseNot_reshape;
			n->operator = BitwiseNot_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = BitwiseNot_init;
			n->exit = BitwiseNot_exit;
			n->reshape = BitwiseNot_reshape;
			n->operator = BitwiseNot_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = BitwiseNot_init;
			n->exit = BitwiseNot_exit;
			n->reshape = BitwiseNot_reshape;
			n->operator = BitwiseNot_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = BitwiseNot_init;
			n->exit = BitwiseNot_exit;
			n->reshape = BitwiseNot_reshape;
			n->operator = BitwiseNot_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = BitwiseNot_init;
			n->exit = BitwiseNot_exit;
			n->reshape = BitwiseNot_reshape;
			n->operator = BitwiseNot_uint64;
			break;
		default:
			break;
		}
	}
}
