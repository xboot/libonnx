#include <onnx.h>

static int GlobalMaxPool_init(struct onnx_node_t * n)
{
	if((n->ninput == 1) && (n->noutput == 1))
		return 1;
	return 0;
}

static int GlobalMaxPool_exit(struct onnx_node_t * n)
{
	return 1;
}

static int GlobalMaxPool_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int ndim = x->ndim;
	int dims[ndim];
	int i;

	for(i = 0; i < ndim; i++)
	{
		if(i < 2)
			dims[i] = x->dims[i];
		else
			dims[i] = 1;
	}
	return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static void GlobalMaxPool_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for(i = 0; i < N; ++i)
	{
		for(j = 0; j < C; ++j)
		{
			o = i * C + j;
			v = float16_to_float32(px[o * m]);
			for(k = 1; k < m; ++k)
				v = fmaxf(v, float16_to_float32(px[o * m + k]));
			py[o] = float32_to_float16(v);
		}
	}
}

static void GlobalMaxPool_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for(i = 0; i < N; ++i)
	{
		for(j = 0; j < C; ++j)
		{
			o = i * C + j;
			py[o] = px[o * m];
			for(k = 1; k < m; ++k)
				py[o] = fmaxf(py[o], px[o * m + k]);
		}
	}
}

static void GlobalMaxPool_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for(i = 0; i < N; ++i)
	{
		for(j = 0; j < C; ++j)
		{
			o = i * C + j;
			py[o] = px[o * m];
			for(k = 1; k < m; ++k)
				py[o] = fmax(py[o], px[o * m + k]);
		}
	}
}

void resolver_default_op_GlobalMaxPool(struct onnx_node_t * n)
{
	switch(n->inputs[0]->type)
	{
	case ONNX_TENSOR_TYPE_FLOAT16:
		n->init = GlobalMaxPool_init;
		n->exit = GlobalMaxPool_exit;
		n->reshape = GlobalMaxPool_reshape;
		n->operator = GlobalMaxPool_float16;
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		n->init = GlobalMaxPool_init;
		n->exit = GlobalMaxPool_exit;
		n->reshape = GlobalMaxPool_reshape;
		n->operator = GlobalMaxPool_float32;
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		n->init = GlobalMaxPool_init;
		n->exit = GlobalMaxPool_exit;
		n->reshape = GlobalMaxPool_reshape;
		n->operator = GlobalMaxPool_float64;
		break;
	default:
		break;
	}
}
