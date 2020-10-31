#include <onnx.h>

static int Mean_init(struct onnx_node_t * n)
{
	if((n->ninput >= 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Mean_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Mean_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	int i;

	if(!onnx_tensor_reshape_identity(y, n->inputs[0], n->inputs[0]->type))
		return 0;
	for(i = 1; i < n->ninput; i++)
	{
		if(!onnx_tensor_reshape_multi_broadcast(y, y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

static void Mean_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float sum;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, sum = 0; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			sum += bfloat16_to_float32(*px);
		}
		py[i] = float32_to_bfloat16(sum / n->ninput);
	}
}

static void Mean_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float sum;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, sum = 0; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			sum += float16_to_float32(*px);
		}
		py[i] = float32_to_float16(sum / n->ninput);
	}
}

static void Mean_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	float * py = (float *)y->datas;
	float * px;
	float sum;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, sum = 0; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			sum += *px;
		}
		py[i] = sum / n->ninput;
	}
}

static void Mean_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x;
	double * py = (double *)y->datas;
	double * px;
	double sum;
	int i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, sum = 0; j < n->ninput; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			sum += *px;
		}
		py[i] = sum / n->ninput;
	}
}

void resolver_default_op_Mean(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_BFLOAT16:
		n->init = Mean_init;
		n->exit = Mean_exit;
		n->reshape = Mean_reshape;
		n->operator = Mean_bfloat16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = Mean_init;
		n->exit = Mean_exit;
		n->reshape = Mean_reshape;
		n->operator = Mean_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = Mean_init;
		n->exit = Mean_exit;
		n->reshape = Mean_reshape;
		n->operator = Mean_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = Mean_init;
		n->exit = Mean_exit;
		n->reshape = Mean_reshape;
		n->operator = Mean_float64;
		break;
	default:
		break;
	}
}
