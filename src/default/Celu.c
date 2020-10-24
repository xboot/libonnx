#include <onnx.h>

struct operator_pdata_t {
	float alpha;
};

static int Celu_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	struct onnx_tensor_t * x;
	struct onnx_tensor_t * y;

	if((n->ninput > 0) && (n->noutput > 0))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			x = n->inputs[0];
			y = n->outputs[0];
			if(!onnx_tensor_shape_equal(y, x) || (y->type != x->type))
				onnx_tensor_reinit(y, x->type, x->dims, x->ndim);
			pdat->alpha = onnx_attribute_read_float(n, "alpha", 1.0);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int Celu_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static void Celu_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = max((float)0.0, (float)px[i]) + min((float)0.0, (float)pdat->alpha * (expf(px[i] / pdat->alpha) - 1));
}

void resolver_default_op_Celu(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = Celu_init;
		n->exit = Celu_exit;
		n->op = Celu_float32;
		break;
	default:
		break;
	}
}
