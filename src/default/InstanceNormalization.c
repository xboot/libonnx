#include <onnx.h>

struct operator_pdata_t {
	float epsilon;
};

static int InstanceNormalization_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 3) && (n->noutput >= 1))
	{
		pdat = onnx_malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->epsilon = onnx_attribute_read_float(n, "epsilon", 1e-05);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int InstanceNormalization_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		onnx_free(pdat);
	return 1;
}

static int InstanceNormalization_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void InstanceNormalization_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * scale = n->inputs[1];
	struct onnx_tensor_t * b = n->inputs[2];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * pscale = (uint16_t *)scale->datas;
	uint16_t * pb = (uint16_t *)b->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float temp, mean, var;
	int N = x->dims[0];
	int C = x->dims[1];
	int NC = N * C;
	int channel = 1;
	int i, j, l, o, jc;

	for(i = 2; i < x->ndim; i++)
		channel *= x->dims[i];
	for(j = 0; j < NC; j++)
	{
		o = j * channel;
		l = o + channel;
		jc = j % C;
		temp = 0;
		for(i = o; i < l; i++)
			temp += float16_to_float32(px[i]);
		mean = temp / channel;
		temp = 0;
		for(i = o; i < l; i++)
			temp += pow(float16_to_float32(px[i]) - mean, 2);
		var = temp / channel;
		for(i = o; i < l; i++)
			py[i] = float32_to_float16(float16_to_float32(pscale[jc]) * ((float16_to_float32(px[i]) - mean) / sqrtf(var + pdat->epsilon)) + float16_to_float32(pb[jc]));
	}
}

static void InstanceNormalization_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * scale = n->inputs[1];
	struct onnx_tensor_t * b = n->inputs[2];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * pscale = (float *)scale->datas;
	float * pb = (float *)b->datas;
	float * py = (float *)y->datas;
	float temp, mean, var;
	int N = x->dims[0];
	int C = x->dims[1];
	int NC = N * C;
	int channel = 1;
	int i, j, l, o, jc;

	for(i = 2; i < x->ndim; i++)
		channel *= x->dims[i];
	for(j = 0; j < NC; j++)
	{
		o = j * channel;
		l = o + channel;
		jc = j % C;
		temp = 0;
		for(i = o; i < l; i++)
			temp += px[i];
		mean = temp / channel;
		temp = 0;
		for(i = o; i < l; i++)
			temp += pow(px[i] - mean, 2);
		var = temp / channel;
		for(i = o; i < l; i++)
			py[i] = pscale[jc] * ((px[i] - mean) / sqrtf(var + pdat->epsilon)) + pb[jc];
	}
}

static void InstanceNormalization_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * scale = n->inputs[1];
	struct onnx_tensor_t * b = n->inputs[2];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * pscale = (double *)scale->datas;
	double * pb = (double *)b->datas;
	double * py = (double *)y->datas;
	double temp, mean, var;
	int N = x->dims[0];
	int C = x->dims[1];
	int NC = N * C;
	int channel = 1;
	int i, j, l, o, jc;

	for(i = 2; i < x->ndim; i++)
		channel *= x->dims[i];
	for(j = 0; j < NC; j++)
	{
		o = j * channel;
		l = o + channel;
		jc = j % C;
		temp = 0;
		for(i = o; i < l; i++)
			temp += px[i];
		mean = temp / channel;
		temp = 0;
		for(i = o; i < l; i++)
			temp += pow(px[i] - mean, 2);
		var = temp / channel;
		for(i = o; i < l; i++)
			py[i] = pscale[jc] * ((px[i] - mean) / sqrt(var + pdat->epsilon)) + pb[jc];
	}
}

void resolver_default_op_InstanceNormalization(struct onnx_node_t * n)
{
	if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = InstanceNormalization_init;
			n->exit = InstanceNormalization_exit;
			n->reshape = InstanceNormalization_reshape;
			n->operator = InstanceNormalization_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = InstanceNormalization_init;
			n->exit = InstanceNormalization_exit;
			n->reshape = InstanceNormalization_reshape;
			n->operator = InstanceNormalization_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = InstanceNormalization_init;
			n->exit = InstanceNormalization_exit;
			n->reshape = InstanceNormalization_reshape;
			n->operator = InstanceNormalization_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = InstanceNormalization_init;
			n->exit = InstanceNormalization_exit;
			n->reshape = InstanceNormalization_reshape;
			n->operator = InstanceNormalization_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = InstanceNormalization_init;
			n->exit = InstanceNormalization_exit;
			n->reshape = InstanceNormalization_reshape;
			n->operator = InstanceNormalization_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = InstanceNormalization_init;
			n->exit = InstanceNormalization_exit;
			n->reshape = InstanceNormalization_reshape;
			n->operator = InstanceNormalization_float64;
			break;
		default:
			break;
		}
	}
}
