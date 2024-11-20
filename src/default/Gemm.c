#include <onnx.h>

struct operator_pdata_t {
	float alpha;
	float beta;
	int transA;
	int transB;

	int m;
	int n;
	int k;
};

static int Gemm_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput >= 2) && (n->noutput == 1))
	{
		pdat = onnx_malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->alpha = onnx_attribute_read_float(n, "alpha", 1.0);
			pdat->beta = onnx_attribute_read_float(n, "beta", 1.0);
			pdat->transA = onnx_attribute_read_int(n, "transA", 0);
			pdat->transB = onnx_attribute_read_int(n, "transB", 0);
			pdat->m = 0;
			pdat->n = 0;
			pdat->k = 0;
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int Gemm_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		onnx_free(pdat);
	return 1;
}

static int Gemm_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	int k;

	if(pdat->transA)
	{
		pdat->m = a->dims[1];
		pdat->k = a->dims[0];
	}
	else
	{
		pdat->m = a->dims[0];
		pdat->k = a->dims[1];
	}
	if(pdat->transB)
	{
		pdat->n = b->dims[0];
		k = 1;
	}
	else
	{
		pdat->n = b->dims[1];
		k = 0;
	}
	if(b->dims[k] != pdat->k)
		return 0;
	if(pdat->m <= 0 || pdat->n <= 0 || pdat->k <= 0)
		return 0;
	if((n->ninput > 2) && !onnx_tensor_broadcast_is_valid(n->inputs[2], (int[]){ pdat->m, pdat->n }, 2))
		return 0;
	return onnx_tensor_reshape(y, (int[]){ pdat->m, pdat->n }, 2, a->type);
}

static void Gemm_int32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	struct onnx_tensor_t * c = (n->ninput > 2) ? n->inputs[2] : NULL;
	int32_t * py = (int32_t *)y->datas;
	int32_t * pa = (int32_t *)a->datas;
	int32_t * pb = (int32_t *)b->datas;
	int32_t * pc;
	int32_t sum;
	int oa = 0;
	int ob = 0;
	int oy = 0;
	int i, j, k;

	if(pdat->transA && pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += 1;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa++;
		}
	}
	else if(pdat->transA)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += pdat->n;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa++;
		}
	}
	else if(pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += 1;
				}
				oa -= pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa += pdat->k;
		}
	}
	else
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += pdat->n;
				}
				oa -= pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa += pdat->k;
		}
	}
}

static void Gemm_int64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	struct onnx_tensor_t * c = (n->ninput > 2) ? n->inputs[2] : NULL;
	int64_t * py = (int64_t *)y->datas;
	int64_t * pa = (int64_t *)a->datas;
	int64_t * pb = (int64_t *)b->datas;
	int64_t * pc;
	int64_t sum;
	int oa = 0;
	int ob = 0;
	int oy = 0;
	int i, j, k;

	if(pdat->transA && pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += 1;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa++;
		}
	}
	else if(pdat->transA)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += pdat->n;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa++;
		}
	}
	else if(pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += 1;
				}
				oa -= pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa += pdat->k;
		}
	}
	else
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += pdat->n;
				}
				oa -= pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa += pdat->k;
		}
	}
}

static void Gemm_uint32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	struct onnx_tensor_t * c = (n->ninput > 2) ? n->inputs[2] : NULL;
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t * pa = (uint32_t *)a->datas;
	uint32_t * pb = (uint32_t *)b->datas;
	uint32_t * pc;
	uint32_t sum;
	int oa = 0;
	int ob = 0;
	int oy = 0;
	int i, j, k;

	if(pdat->transA && pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += 1;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa++;
		}
	}
	else if(pdat->transA)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += pdat->n;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa++;
		}
	}
	else if(pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += 1;
				}
				oa -= pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa += pdat->k;
		}
	}
	else
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += pdat->n;
				}
				oa -= pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa += pdat->k;
		}
	}
}

static void Gemm_uint64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	struct onnx_tensor_t * c = (n->ninput > 2) ? n->inputs[2] : NULL;
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * pa = (uint64_t *)a->datas;
	uint64_t * pb = (uint64_t *)b->datas;
	uint64_t * pc;
	uint64_t sum;
	int oa = 0;
	int ob = 0;
	int oy = 0;
	int i, j, k;

	if(pdat->transA && pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += 1;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa++;
		}
	}
	else if(pdat->transA)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += pdat->n;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa++;
		}
	}
	else if(pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += 1;
				}
				oa -= pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa += pdat->k;
		}
	}
	else
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += pdat->n;
				}
				oa -= pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa += pdat->k;
		}
	}
}

static void Gemm_bfloat16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	struct onnx_tensor_t * c = (n->ninput > 2) ? n->inputs[2] : NULL;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * pa = (uint16_t *)a->datas;
	uint16_t * pb = (uint16_t *)b->datas;
	uint16_t * pc;
	float sum;
	int oa = 0;
	int ob = 0;
	int oy = 0;
	int i, j, k;

	if(pdat->transA && pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += bfloat16_to_float32(pa[oa]) * bfloat16_to_float32(pb[ob]);
					oa += pdat->m;
					ob += 1;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = float32_to_bfloat16(pdat->alpha * sum + pdat->beta * bfloat16_to_float32(*pc));
				}
				else
					py[oy] = float32_to_bfloat16(pdat->alpha * sum);
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa++;
		}
	}
	else if(pdat->transA)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += bfloat16_to_float32(pa[oa]) * bfloat16_to_float32(pb[ob]);
					oa += pdat->m;
					ob += pdat->n;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = float32_to_bfloat16(pdat->alpha * sum + pdat->beta * bfloat16_to_float32(*pc));
				}
				else
					py[oy] = float32_to_bfloat16(pdat->alpha * sum);
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa++;
		}
	}
	else if(pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += bfloat16_to_float32(pa[oa]) * bfloat16_to_float32(pb[ob]);
					oa += 1;
					ob += 1;
				}
				oa -= pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = float32_to_bfloat16(pdat->alpha * sum + pdat->beta * bfloat16_to_float32(*pc));
				}
				else
					py[oy] = float32_to_bfloat16(pdat->alpha * sum);
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa += pdat->k;
		}
	}
	else
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += bfloat16_to_float32(pa[oa]) * bfloat16_to_float32(pb[ob]);
					oa += 1;
					ob += pdat->n;
				}
				oa -= pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = float32_to_bfloat16(pdat->alpha * sum + pdat->beta * bfloat16_to_float32(*pc));
				}
				else
					py[oy] = float32_to_bfloat16(pdat->alpha * sum);
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa += pdat->k;
		}
	}
}

static void Gemm_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	struct onnx_tensor_t * c = (n->ninput > 2) ? n->inputs[2] : NULL;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * pa = (uint16_t *)a->datas;
	uint16_t * pb = (uint16_t *)b->datas;
	uint16_t * pc;
	float sum;
	int oa = 0;
	int ob = 0;
	int oy = 0;
	int i, j, k;

	if(pdat->transA && pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += float16_to_float32(pa[oa]) * float16_to_float32(pb[ob]);
					oa += pdat->m;
					ob += 1;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = float32_to_float16(pdat->alpha * sum + pdat->beta * float16_to_float32(*pc));
				}
				else
					py[oy] = float32_to_float16(pdat->alpha * sum);
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa++;
		}
	}
	else if(pdat->transA)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += float16_to_float32(pa[oa]) * float16_to_float32(pb[ob]);
					oa += pdat->m;
					ob += pdat->n;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = float32_to_float16(pdat->alpha * sum + pdat->beta * float16_to_float32(*pc));
				}
				else
					py[oy] = float32_to_float16(pdat->alpha * sum);
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa++;
		}
	}
	else if(pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += float16_to_float32(pa[oa]) * float16_to_float32(pb[ob]);
					oa += 1;
					ob += 1;
				}
				oa -= pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = float32_to_float16(pdat->alpha * sum + pdat->beta * float16_to_float32(*pc));
				}
				else
					py[oy] = float32_to_float16(pdat->alpha * sum);
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa += pdat->k;
		}
	}
	else
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += float16_to_float32(pa[oa]) * float16_to_float32(pb[ob]);
					oa += 1;
					ob += pdat->n;
				}
				oa -= pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = float32_to_float16(pdat->alpha * sum + pdat->beta * float16_to_float32(*pc));
				}
				else
					py[oy] = float32_to_float16(pdat->alpha * sum);
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa += pdat->k;
		}
	}
}

static void Gemm_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	struct onnx_tensor_t * c = (n->ninput > 2) ? n->inputs[2] : NULL;
	float * py = (float *)y->datas;
	float * pa = (float *)a->datas;
	float * pb = (float *)b->datas;
	float * pc;
	float sum;
	int oa = 0;
	int ob = 0;
	int oy = 0;
	int i, j, k;

	if(pdat->transA && pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += 1;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa++;
		}
	}
	else if(pdat->transA)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += pdat->n;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa++;
		}
	}
	else if(pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += 1;
				}
				oa -= pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa += pdat->k;
		}
	}
	else
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += pdat->n;
				}
				oa -= pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa += pdat->k;
		}
	}
}

static void Gemm_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	struct onnx_tensor_t * c = (n->ninput > 2) ? n->inputs[2] : NULL;
	double * py = (double *)y->datas;
	double * pa = (double *)a->datas;
	double * pb = (double *)b->datas;
	double * pc;
	double sum;
	int oa = 0;
	int ob = 0;
	int oy = 0;
	int i, j, k;

	if(pdat->transA && pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += 1;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa++;
		}
	}
	else if(pdat->transA)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += pdat->n;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa++;
		}
	}
	else if(pdat->transB)
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += 1;
				}
				oa -= pdat->k;
				ob -= pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa += pdat->k;
		}
	}
	else
	{
		for(i = 0; i < pdat->m; i++)
		{
			for(j = 0; j < pdat->n; j++)
			{
				sum = 0;
				for(k = 0; k < pdat->k; k++)
				{
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += pdat->n;
				}
				oa -= pdat->k;
				ob -= pdat->n * pdat->k;
				if(c)
				{
					pc = onnx_tensor_broadcast_map_address(c, y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa += pdat->k;
		}
	}
}

void resolver_default_op_Gemm(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 11)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 9)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Gemm_init;
			n->exit = Gemm_exit;
			n->reshape = Gemm_reshape;
			n->operator = Gemm_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 6)
	{
	}
	else if(n->opset >= 1)
	{
	}
}
