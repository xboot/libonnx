#include <onnx.h>

struct operator_pdata_t {
	float alpha;
};

static void LeakyRelu_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	Onnx__TensorProto * t = n->inputs[0];
	Onnx__AttributeProto * a;
	int i;

	for(i = 0; i < n->noutput; i++)
	{
		if(n->outputs[i]->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED)
			onnx_tensor_ready(n->outputs[i], t->data_type, t->n_dims, t->dims);
	}

	pdat = malloc(sizeof(struct operator_pdata_t));
	if(pdat)
	{
		a = onnx_search_attribute(n, "alpha");
		pdat->alpha = onnx_attribute_read_float(a, 0.01);
	}
	n->priv = pdat;
}

static void LeakyRelu_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
}

static void LeakyRelu_float(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	Onnx__TensorProto * x = n->inputs[0];
	Onnx__TensorProto * y = n->outputs[0];
	int i, l;

	for(i = 0, l = y->n_float_data; i < l; i++)
		y->float_data[i] = (x->float_data[i] < 0) ? x->float_data[i] * pdat->alpha : x->float_data[i];
}

static void LeakyRelu_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	Onnx__TensorProto * x = n->inputs[0];
	Onnx__TensorProto * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->int32_data;
	uint16_t * py = (uint16_t *)y->int32_data;
	float v;
	int i, l;

	for(i = 0, l = (y->n_int32_data << 1); i < l; i++)
	{
		v = float16_to_float32(px[i]);
		if(v < 0)
			v *= pdat->alpha;
		py[i] = float32_to_float16(v);
	}
}

static void LeakyRelu_double(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	Onnx__TensorProto * x = n->inputs[0];
	Onnx__TensorProto * y = n->outputs[0];
	int i, l;

	for(i = 0, l = y->n_double_data; i < l; i++)
		y->double_data[i] = (x->double_data[i] < 0) ? x->double_data[i] * pdat->alpha : x->double_data[i];
}

void default_resolver_op_LeakyRelu(struct onnx_node_t * n)
{
	switch(n->inputs[0]->data_type)
	{
	case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
		n->init = LeakyRelu_init;
		n->exit = LeakyRelu_exit;
		n->op = LeakyRelu_float;
		break;
	case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
		n->init = LeakyRelu_init;
		n->exit = LeakyRelu_exit;
		n->op = LeakyRelu_float16;
		break;
	case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
		n->init = LeakyRelu_init;
		n->exit = LeakyRelu_exit;
		n->op = LeakyRelu_double;
		break;
	default:
		break;
	}
}
