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
#include <default/default.h>

#define ONNX_LOG(...)	printf(__VA_ARGS__)

static void hmap_entry_callback(struct hmap_entry_t * e)
{
	if(e && e->value)
		onnx_tensor_free((struct onnx_tensor_t *)e->value);
}

struct onnx_context_t * onnx_context_alloc(const void * buf, size_t len, struct onnx_resolver_t ** r, int rlen)
{
	struct onnx_context_t * ctx;
	int i;

	if(!buf || len <= 0)
		return NULL;

	ctx = malloc(sizeof(struct onnx_context_t));
	if(!ctx)
		return NULL;

	ctx->model = onnx__model_proto__unpack(NULL, len, buf);
	if(!ctx->model)
	{
		if(ctx)
			free(ctx);
		return NULL;
	}

	ctx->map = hmap_alloc(0);
	if(!ctx->map)
	{
		if(ctx->model)
			onnx__model_proto__free_unpacked(ctx->model, NULL);
		if(ctx)
			free(ctx);
		return NULL;
	}

	ctx->rlen = rlen;
	if(r && (ctx->rlen > 0))
	{
		ctx->r = malloc(sizeof(struct onnx_resolver_t *) * ctx->rlen);
		ctx->rctx = malloc(sizeof(void *) * ctx->rlen);
		if(!ctx->r || !ctx->rctx)
		{
			if(ctx->rctx)
				free(ctx->rctx);
			if(ctx->r)
				free(ctx->r);
			if(ctx->map)
				hmap_free(ctx->map, hmap_entry_callback);
			if(ctx->model)
				onnx__model_proto__free_unpacked(ctx->model, NULL);
			if(ctx)
				free(ctx);
			return NULL;
		}
	}
	else
	{
		ctx->r = NULL;
		ctx->rctx = NULL;
	}

	for(i = 0; i < ctx->rlen; i++)
	{
		ctx->r[i] = r[i];
		if(r[i] && r[i]->create)
			ctx->rctx[i] = r[i]->create();
	}

	ctx->g = onnx_graph_alloc(ctx, ctx->model->graph);
	if(!ctx->g)
	{
		for(i = 0; i < ctx->rlen; i++)
		{
			if(ctx->r[i] && ctx->r[i]->destroy)
				ctx->r[i]->destroy(ctx->rctx[i]);
		}
		if(ctx->rctx)
			free(ctx->rctx);
		if(ctx->r)
			free(ctx->r);
		if(ctx->map)
			hmap_free(ctx->map, hmap_entry_callback);
		if(ctx->model)
			onnx__model_proto__free_unpacked(ctx->model, NULL);
		if(ctx)
			free(ctx);
		return NULL;
	}

	return ctx;
}

struct onnx_context_t * onnx_context_alloc_from_file(const char * filename, struct onnx_resolver_t ** r, int rlen)
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
				ctx = onnx_context_alloc(buf, len, r, rlen);
				free(buf);
			}
		}
		fclose(fp);
	}
	return ctx;
}

void onnx_context_free(struct onnx_context_t * ctx)
{
	int i;

	if(ctx)
	{
		if(ctx->g)
			onnx_graph_free(ctx->g);
		for(i = 0; i < ctx->rlen; i++)
		{
			if(ctx->r[i] && ctx->r[i]->destroy)
				ctx->r[i]->destroy(ctx->rctx[i]);
		}
		if(ctx->rctx)
			free(ctx->rctx);
		if(ctx->r)
			free(ctx->r);
		if(ctx->map)
			hmap_free(ctx->map, hmap_entry_callback);
		if(ctx->model)
			onnx__model_proto__free_unpacked(ctx->model, NULL);
		free(ctx);
	}
}

static struct onnx_tensor_t * onnx_tensor_alloc_from_value_info(Onnx__ValueInfoProto * v)
{
	struct onnx_tensor_t * t;
	enum onnx_tensor_type_t type;
	int * dims = NULL;
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
			dims = malloc(sizeof(int) * ndim);
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
						if(strcmp(v->type->tensor_type->shape->dim[i]->dim_param, "batch_size") == 0)
							dims[i] = 1;
						else
							dims[i] = 1;
						break;
					default:
						dims[i] = 1;
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
	size_t n, i;
	int sz;

	if(t && o)
	{
		if(t->type == o->data_type)
		{
			sz = onnx_tensor_type_sizeof(t->type);
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
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
								{
									v.u = le32_to_cpu(q[i]);
									p[i] = v.f;
								}
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
						{
							uint8_t * p = (uint8_t *)t->datas;
							uint8_t * q = (uint8_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len);
								memcpy(p, q, n);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
						{
							int8_t * p = (int8_t *)t->datas;
							int8_t * q = (int8_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len);
								memcpy(p, q, n);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
						{
							uint16_t * p = (uint16_t *)t->datas;
							uint16_t * q = (uint16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
						{
							int16_t * p = (int16_t *)t->datas;
							int16_t * q = (int16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
						{
							int32_t * p = (int32_t *)t->datas;
							int32_t * q = (int32_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le32_to_cpu(q[i]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
						{
							int64_t * p = (int64_t *)t->datas;
							int64_t * q = (int64_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le64_to_cpu(q[i]);
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
								n = min(t->ndata, (size_t)o->raw_data.len);
								memcpy(p, q, n);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
						{
							uint16_t * p = (uint16_t *)t->datas;
							uint16_t * q = (uint16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
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
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
								{
									v.u = le64_to_cpu(q[i]);
									p[i] = v.f;
								}
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
						{
							uint32_t * p = (uint32_t *)t->datas;
							uint32_t * q = (uint32_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le32_to_cpu(q[i]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
						{
							uint64_t * p = (uint64_t *)t->datas;
							uint64_t * q = (uint64_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le64_to_cpu(q[i]);
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
								n = min(t->ndata, (size_t)o->raw_data.len / sz) * 2;
								for(i = 0; i < n; i++)
								{
									v.u = le32_to_cpu(q[i]);
									p[i] = v.f;
								}
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
								n = min(t->ndata, (size_t)o->raw_data.len / sz) * 2;
								for(i = 0; i < n; i++)
								{
									v.u = le64_to_cpu(q[i]);
									p[i] = v.f;
								}
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
						{
							uint16_t * p = (uint16_t *)t->datas;
							uint16_t * q = (uint16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
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
						n = min(t->ndata, (size_t)o->n_float_data);
						if((n > 0) && t->datas && o->float_data)
							memcpy(t->datas, o->float_data, sizeof(float) * n);
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
						n = min(t->ndata, (size_t)o->n_int32_data);
						if((n > 0) && t->datas && o->int32_data)
							memcpy(t->datas, o->int32_data, sz * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
						n = min(t->ndata, (size_t)o->n_string_data);
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
						n = min(t->ndata, (size_t)o->n_int64_data);
						if((n > 0) && t->datas && o->int64_data)
							memcpy(t->datas, o->int64_data, sizeof(int64_t) * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
						n = min(t->ndata, (size_t)o->n_double_data);
						if((n > 0) && t->datas && o->double_data)
							memcpy(t->datas, o->double_data, sizeof(double) * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
						//TODO
						n = min(t->ndata, (size_t)o->n_uint64_data);
						if((n > 0) && t->datas && o->uint64_data)
							memcpy(t->datas, o->uint64_data, sz * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
						n = min(t->ndata, (size_t)(o->n_float_data / 2));
						if((n > 0) && t->datas && o->float_data)
							memcpy(t->datas, o->float_data, sizeof(float) * 2 * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
						n = min(t->ndata, (size_t)(o->n_double_data / 2));
						if((n > 0) && t->datas && o->double_data)
							memcpy(t->datas, o->double_data, sizeof(double) * 2 * n);
						break;
					default:
						break;
					}
				}
			}
		}
	}
}

static int reshape_dummy(struct onnx_node_t * n)
{
	return 1;
}

static void operator_dummy(struct onnx_node_t * n)
{
	ONNX_LOG("\033[45;37mUnsupported opset\033[0m => %s-%d (%s)\r\n", n->proto->op_type, n->opset, (strlen(n->proto->domain) > 0) ? n->proto->domain : "ai.onnx");
}

static void resolver_solve_operator(struct onnx_resolver_t * r, struct onnx_node_t * n)
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
		case 0xe6ece5fb: /* "Scatter" */
			rop = r->op_Scatter;
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
		case 0xae63c66c: /* "Upsample" */
			rop = r->op_Upsample;
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

struct onnx_graph_t * onnx_graph_alloc(struct onnx_context_t * ctx, Onnx__GraphProto * graph)
{
	struct onnx_graph_t * g;
	struct onnx_node_t * n;
	struct onnx_tensor_t * t;
	Onnx__TensorProto * o;
	Onnx__ValueInfoProto * v;
	char * p, * domain;
	char * name;
	int i, j, k, l;

	if(!graph)
		return NULL;

	g = malloc(sizeof(struct onnx_graph_t));
	if(!g)
		return NULL;
	memset(g, 0, sizeof(struct onnx_graph_t));

	g->nlen = graph->n_node;
	g->nodes = malloc(sizeof(struct onnx_node_t) * g->nlen);
	if(!g->nodes)
	{
		free(g);
		return NULL;
	}

	for(i = 0; i < graph->n_input; i++)
	{
		v = graph->input[i];
		if(!onnx_tensor_search(ctx, v->name))
		{
			t = onnx_tensor_alloc_from_value_info(v);
			if(t)
			{
				for(j = 0; j < graph->n_initializer; j++)
				{
					if(strcmp(graph->initializer[j]->name, t->name) == 0)
					{
						onnx_tensor_copy_from_tensor_proto(t, graph->initializer[j]);
						break;
					}
				}
				hmap_add(ctx->map, t->name, t);
			}
		}
	}

	for(i = 0; i < graph->n_output; i++)
	{
		v = graph->output[i];
		if(!onnx_tensor_search(ctx, v->name))
		{
			t = onnx_tensor_alloc_from_value_info(v);
			if(t)
				hmap_add(ctx->map, t->name, t);
		}
	}

	for(i = 0; i < graph->n_value_info; i++)
	{
		v = graph->value_info[i];
		if(!onnx_tensor_search(ctx, v->name))
		{
			t = onnx_tensor_alloc_from_value_info(v);
			if(t)
				hmap_add(ctx->map, t->name, t);
		}
	}

	for(i = 0; i < graph->n_node; i++)
	{
		for(j = 0; j < graph->node[i]->n_output; j++)
		{
			name = graph->node[i]->output[j];
			if(!onnx_tensor_search(ctx, name))
			{
				t = onnx_tensor_alloc(name, ONNX_TENSOR_TYPE_UNDEFINED, NULL, 0);
				if(t)
					hmap_add(ctx->map, name, t);
			}
		}
	}

	for(i = 0; i < graph->n_node; i++)
	{
		for(j = 0; j < graph->node[i]->n_input; j++)
		{
			name = graph->node[i]->input[j];
			if(!onnx_tensor_search(ctx, name))
			{
				for(k = 0; k < graph->n_initializer; k++)
				{
					if(strcmp(graph->initializer[k]->name, name) == 0)
					{
						o = graph->initializer[k];
						if(o)
						{
							int ndim = o->n_dims;
							int dims[ndim];
							for(l = 0; l < ndim; l++)
								dims[l] = o->dims[l];
							t = onnx_tensor_alloc(name, (enum onnx_tensor_type_t)o->data_type, dims, ndim);
							if(t)
							{
								onnx_tensor_copy_from_tensor_proto(t, o);
								hmap_add(ctx->map, name, t);
							}
							break;
						}
					}
				}
				if(!onnx_tensor_search(ctx, name))
				{
					if(g->nodes)
						free(g->nodes);
					free(g);
					return NULL;
				}
			}
		}
	}

	for(i = 0; i < g->nlen; i++)
	{
		n = &g->nodes[i];
		memset(n, 0, sizeof(struct onnx_node_t));

		n->ctx = ctx;
		n->proto = graph->node[i];
		domain = n->proto->domain;
		if(!domain || (strlen(domain) == 0))
			domain = "ai.onnx";
		for(j = 0; j < ctx->model->n_opset_import; j++)
		{
			p = ctx->model->opset_import[j]->domain;
			if(!p || (strlen(p) == 0))
				p = "ai.onnx";
			if(strcmp(domain, p) == 0)
			{
				n->opset = ctx->model->opset_import[j]->version;
				break;
			}
		}
		if(n->proto->n_input > 0)
		{
			n->inputs = malloc(sizeof(struct onnx_tensor_t *) * n->proto->n_input);
			if(n->inputs)
			{
				n->ninput = n->proto->n_input;
				for(j = 0; j < n->ninput; j++)
					n->inputs[j] = onnx_tensor_search(ctx, n->proto->input[j]);
			}
		}
		if(n->proto->n_output > 0)
		{
			n->outputs = malloc(sizeof(struct onnx_tensor_t *) * n->proto->n_output);
			if(n->outputs)
			{
				n->noutput = n->proto->n_output;
				for(j = 0; j < n->noutput; j++)
					n->outputs[j] = onnx_tensor_search(ctx, n->proto->output[j]);
			}
		}
		for(j = 0; j < ctx->rlen; j++)
		{
			resolver_solve_operator(ctx->r[j], n);
			if(n->operator)
			{
				n->r = ctx->r[j];
				n->rctx = ctx->rctx[j];
				break;
			}
		}
		if(!n->operator)
		{
			resolver_solve_operator(&resolver_default, n);
			if(n->operator)
			{
				n->r = &resolver_default;
				n->rctx = NULL;
			}
		}
		if(!n->reshape)
			n->reshape = reshape_dummy;
		if(!n->operator)
			n->operator = operator_dummy;
		if(n->init)
		{
			if(n->init(n) <= 0)
			{
				if(g->nodes)
				{
					for(j = 0; j < g->nlen; j++)
					{
						n = &g->nodes[j];
						if(n->exit)
							n->exit(n);
						if(n->inputs)
							free(n->inputs);
						if(n->outputs)
							free(n->outputs);
					}
					free(g->nodes);
				}
				free(g);
				return NULL;
			}
		}
		if(n->reshape)
			n->reshape(n);
	}

	return g;
}

void onnx_graph_free(struct onnx_graph_t * g)
{
	struct onnx_node_t * n;
	int i;

	if(g)
	{
		if(g->nodes)
		{
			for(i = 0; i < g->nlen; i++)
			{
				n = &g->nodes[i];
				if(n->exit)
					n->exit(n);
				if(n->inputs)
					free(n->inputs);
				if(n->outputs)
					free(n->outputs);
			}
			free(g->nodes);
		}
		free(g);
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

int onnx_tensor_type_sizeof(enum onnx_tensor_type_t type)
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

struct onnx_tensor_t * onnx_tensor_search(struct onnx_context_t * ctx, const char * name)
{
	if(ctx)
		return hmap_search(ctx->map, name);
	return NULL;
}

struct onnx_tensor_t * onnx_tensor_alloc(const char * name, enum onnx_tensor_type_t type, int * dims, int ndim)
{
	struct onnx_tensor_t * t;

	if(!name)
		return NULL;

	t = malloc(sizeof(struct onnx_tensor_t));
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
	int * dims = NULL;
	int ndim = 0;
	int i;

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
					if(pb->n_dims > 0)
					{
						dims = malloc(sizeof(int) * pb->n_dims);
						if(dims)
						{
							for(i = 0; i < pb->n_dims; i++)
								dims[i] = pb->dims[i];
							ndim = pb->n_dims;
						}
					}
					t = onnx_tensor_alloc(pb->name, (enum onnx_tensor_type_t)pb->data_type, dims, ndim);
					if((ndim > 0) && dims)
						free(dims);
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

	if(t)
	{
		if(t->name)
			free(t->name);
		if(t->ndim > 0)
		{
			if(t->strides)
				free(t->strides);
			if(t->dims)
				free(t->dims);
		}
		if((t->ndata > 0) && t->datas)
		{
			if(t->type == ONNX_TENSOR_TYPE_STRING)
			{
				str = (char **)t->datas;
				for(size_t idx = 0; idx < t->ndata; idx++)
				{
					if(str[idx])
						free(str[idx]);
				}
			}
			free(t->datas);
		}
		free(t);
	}
}

int onnx_tensor_equal(struct onnx_tensor_t * a, struct onnx_tensor_t * b)
{
	size_t i;

	if(!a || !b)
		return 0;
	if(a->type != b->type)
		return 0;
	if(a->ndim != b->ndim)
		return 0;
	if(a->ndata != b->ndata)
		return 0;
	if(a->ndim > 0)
	{
		if(memcmp(a->dims, b->dims, sizeof(int) * a->ndim) != 0)
			return 0;
	}
	switch(a->type)
	{
	case ONNX_TENSOR_TYPE_BOOL:
	case ONNX_TENSOR_TYPE_INT8:
	case ONNX_TENSOR_TYPE_INT16:
	case ONNX_TENSOR_TYPE_INT32:
	case ONNX_TENSOR_TYPE_INT64:
	case ONNX_TENSOR_TYPE_UINT8:
	case ONNX_TENSOR_TYPE_UINT16:
	case ONNX_TENSOR_TYPE_UINT32:
	case ONNX_TENSOR_TYPE_UINT64:
		if(memcmp(a->datas, b->datas, a->ndata * onnx_tensor_type_sizeof(a->type)) != 0)
			return 0;
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * p = (uint16_t *)a->datas;
			uint16_t * q = (uint16_t *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabsf(bfloat16_to_float32(p[i]) - bfloat16_to_float32(q[i])) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * p = (uint16_t *)a->datas;
			uint16_t * q = (uint16_t *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabsf(float16_to_float32(p[i]) - float16_to_float32(q[i])) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * p = (float *)a->datas;
			float * q = (float *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabsf(p[i] - q[i]) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * p = (double *)a->datas;
			double * q = (double *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabs(p[i] - q[i]) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_COMPLEX64:
		{
			float * p = (float *)a->datas;
			float * q = (float *)b->datas;
			for(i = 0; i < a->ndata * 2; i++)
			{
				if(fabsf(p[i] - q[i]) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_COMPLEX128:
		{
			double * p = (double *)a->datas;
			double * q = (double *)b->datas;
			for(i = 0; i < a->ndata * 2; i++)
			{
				if(fabs(p[i] - q[i]) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** p = (char **)a->datas;
			char ** q = (char **)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(p[i] && q[i] && (strcmp(p[i], q[i]) != 0))
					return 0;
			}
		}
		break;
	default:
		break;
	}
	return 1;
}

void onnx_tensor_reinit(struct onnx_tensor_t * t, enum onnx_tensor_type_t type, int * dims, int ndim)
{
	char ** str;
	size_t n;
	int sz, i;

	if(t)
	{
		if(t->ndim > 0)
		{
			if(t->strides)
			{
				free(t->strides);
				t->strides = NULL;
			}
			if(t->dims)
			{
				free(t->dims);
				t->dims = NULL;
			}
			t->ndim = 0;
		}
		if((t->ndata > 0) && t->datas)
		{
			if(t->type == ONNX_TENSOR_TYPE_STRING)
			{
				str = (char **)t->datas;
				for(size_t idx = 0; idx < t->ndata; idx++)
				{
					if(str[idx])
					{
						free(str[idx]);
						str[idx] = NULL;
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
				for(i = 0; i < ndim; i++)
				{
					if(dims[i] <= 0)
						return;
				}
				t->strides = malloc(sizeof(int) * ndim);
				t->dims = malloc(sizeof(int) * ndim);
				if(t->strides && t->dims)
				{
					t->strides[ndim - 1] = 1;
					for(i = ndim - 2; i >= 0; i--)
						t->strides[i] = dims[i + 1] * t->strides[i + 1];
					memcpy(t->dims, dims, sizeof(int) * ndim);
					t->ndim = ndim;
					for(i = 0, n = 1; i < t->ndim; i++)
						n *= t->dims[i];
					sz = onnx_tensor_type_sizeof(t->type);
					if(sz > 0)
					{
						t->datas = memalign(512, n * sz);
						if(t->datas)
						{
							memset(t->datas, 0, n * sz);
							t->ndata = n;
						}
					}
				}
				else
				{
					if(t->strides)
					{
						free(t->strides);
						t->strides = NULL;
					}
					if(t->dims)
					{
						free(t->dims);
						t->dims = NULL;
					}
				}
			}
			else
			{
				sz = onnx_tensor_type_sizeof(t->type);
				if(sz > 0)
				{
					t->datas = memalign(512, sz);
					if(t->datas)
					{
						memset(t->datas, 0, sz);
						t->ndata = 1;
					}
				}
			}
		}
	}
}

void onnx_tensor_apply(struct onnx_tensor_t * t, void * buf, size_t len)
{
	size_t l;
	int sz;

	if(t)
	{
		if(t->datas && buf && (len > 0))
		{
			sz = onnx_tensor_type_sizeof(t->type);
			if(sz > 0)
			{
				if(t->type == ONNX_TENSOR_TYPE_STRING)
				{
					char ** p = (char **)t->datas;
					char ** q = (char **)buf;
					for(size_t idx = 0; idx < t->ndata; idx++)
					{
						if(p[idx])
						{
							free(p[idx]);
							p[idx] = NULL;
						}
					}
					l = min(t->ndata, (size_t)len);
					for(size_t idx = 0; idx < l; idx++)
						p[idx] = strdup(q[idx]);
				}
				else
				{
					l = t->ndata * sz;
					if(l > 0)
						memcpy(t->datas, buf, min(l, len));
				}
			}
		}
	}
}

static Onnx__AttributeProto * onnx_search_attribute(struct onnx_node_t * n, const char * name)
{
	Onnx__AttributeProto * attr;
	int i;

	if(n && name)
	{
		for(i = 0; i < n->proto->n_attribute; i++)
		{
			attr = n->proto->attribute[i];
			if(strcmp(attr->name, name) == 0)
				return attr;
		}
	}
	return NULL;
}

float onnx_attribute_read_float(struct onnx_node_t * n, const char * name, float def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT))
		return attr->f;
	return def;
}

int64_t onnx_attribute_read_int(struct onnx_node_t * n, const char * name, int64_t def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT))
		return attr->i;
	return def;
}

char * onnx_attribute_read_string(struct onnx_node_t * n, const char * name, char * def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING))
	{
		if(attr->s.len > 0)
		{
			attr->s.data[attr->s.len] = 0;
			return (char *)attr->s.data;
		}
	}
	return def;
}

int onnx_attribute_read_floats(struct onnx_node_t * n, const char * name, float ** floats)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS))
	{
		*floats = attr->floats;
		return attr->n_floats;
	}
	return 0;
}

int onnx_attribute_read_ints(struct onnx_node_t * n, const char * name, int64_t ** ints)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS))
	{
		*ints = attr->ints;
		return attr->n_ints;
	}
	return 0;
}

int onnx_attribute_read_tensor(struct onnx_node_t * n, const char * name, struct onnx_tensor_t * t)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);
	int * dims = NULL;
	int ndim = 0;
	int i;

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR))
	{
		if(attr->t)
		{
			if(attr->t->n_dims > 0)
			{
				dims = malloc(sizeof(int) * attr->t->n_dims);
				if(dims)
				{
					for(i = 0; i < attr->t->n_dims; i++)
						dims[i] = attr->t->dims[i];
					ndim = attr->t->n_dims;
				}
			}
			if((t->ndim != ndim) || (memcmp(t->dims, dims, sizeof(int) * ndim) != 0) || (t->type != (enum onnx_tensor_type_t)attr->t->data_type))
				onnx_tensor_reinit(t, (enum onnx_tensor_type_t)attr->t->data_type, dims, ndim);
			if((ndim > 0) && dims)
				free(dims);
			onnx_tensor_copy_from_tensor_proto(t, attr->t);
			return 1;
		}
	}
	return 0;
}

Onnx__GraphProto * onnx_attribute_read_graph(struct onnx_node_t * n, const char * name, Onnx__GraphProto * def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH))
	{
		if(attr->g)
			return attr->g;
	}
	return def;
}

Onnx__SparseTensorProto * onnx_attribute_read_sparse_tensor(struct onnx_node_t * n, const char * name, Onnx__SparseTensorProto * def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR))
	{
		if(attr->sparse_tensor)
			return attr->sparse_tensor;
	}
	return def;
}

void onnx_tensor_dump(struct onnx_tensor_t * t, int detail)
{
	int * sizes, * levels;
	char * lbuf, * rbuf;
	char * lp, * rp;
	void * p;
	int i, j, k;

	if(t)
	{
		ONNX_LOG("%s: %s", t->name, onnx_tensor_type_tostring(t->type));
		if(t->ndim > 0)
		{
			ONNX_LOG("[");
			for(i = 0; i < t->ndim; i++)
			{
				ONNX_LOG("%d", t->dims[i]);
				if(i != t->ndim - 1)
					ONNX_LOG(" x ");
			}
			ONNX_LOG("]");
			if(detail)
			{
				ONNX_LOG(" = \r\n");
				for(i = 0; i < t->ndim; i++)
				{
					if(t->dims[i] <= 0)
						return;
				}
				sizes = malloc(sizeof(int) * t->ndim);
				levels = malloc(sizeof(int) * t->ndim);
				sizes[t->ndim - 1] = t->dims[t->ndim - 1];
				levels[t->ndim - 1] = 0;
				lbuf = malloc(sizeof(char) * (t->ndim + 1));
				rbuf = malloc(sizeof(char) * (t->ndim + 1));
				lp = lbuf;
				rp = rbuf;
				for(i = t->ndim - 2; i >= 0; i--)
				{
					sizes[i] = t->dims[i] * sizes[i + 1];
					levels[i] = 0;
				}
				for(size_t idx = 0; idx < t->ndata; idx++)
				{
					for(j = 0; j < t->ndim; j++)
					{
						if((idx % sizes[j]) == 0)
							levels[j]++;
						if(levels[j] == 1)
						{
							*lp++ = '[';
							levels[j]++;
						}
						if(levels[j] == 3)
						{
							*rp++ = ']';
							if((j != 0) && (levels[j] > levels[j - 1]))
							{
								*lp++ = '[';
								levels[j] = 2;
							}
							else
							{
								levels[j] = 0;
							}
						}
					}
					*lp = *rp = '\0';
					ONNX_LOG("%s", rbuf);
					if(*rbuf != '\0')
					{
						ONNX_LOG("\r\n");
						for(k = t->ndim - strlen(rbuf); k > 0; k--)
							ONNX_LOG(" ");
					}
					ONNX_LOG("%s", lbuf);
					if(*lbuf == '\0')
						ONNX_LOG(" ");
					p = (void *)(t->datas + onnx_tensor_type_sizeof(t->type) * idx);
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
						ONNX_LOG("%s,", (char *)(((char **)p)[0]));
						break;
					default:
						ONNX_LOG("?,");
						break;
					}
					lp = lbuf;
					rp = rbuf;
				}
				for(j = 0; j < t->ndim; j++)
					ONNX_LOG("]");
				free(sizes);
				free(levels);
				free(lbuf);
				free(rbuf);
				ONNX_LOG("\r\n");
			}
			else
			{
				ONNX_LOG(" = ");
				ONNX_LOG("[...]");
				ONNX_LOG("\r\n");
			}
		}
		else if(t->ndata == 1)
		{
			ONNX_LOG(" = ");
			p = (void *)(t->datas);
			switch(t->type)
			{
			case ONNX_TENSOR_TYPE_BOOL:
				ONNX_LOG("%s", *((uint8_t *)p) ? "true" : "false");
				break;
			case ONNX_TENSOR_TYPE_INT8:
				ONNX_LOG("%d", *((int8_t *)p));
				break;
			case ONNX_TENSOR_TYPE_INT16:
				ONNX_LOG("%d", *((int16_t *)p));
				break;
			case ONNX_TENSOR_TYPE_INT32:
				ONNX_LOG("%d", *((int32_t *)p));
				break;
			case ONNX_TENSOR_TYPE_INT64:
				ONNX_LOG("%ld", *((int64_t *)p));
				break;
			case ONNX_TENSOR_TYPE_UINT8:
				ONNX_LOG("%u", *((uint8_t *)p));
				break;
			case ONNX_TENSOR_TYPE_UINT16:
				ONNX_LOG("%u", *((uint16_t *)p));
				break;
			case ONNX_TENSOR_TYPE_UINT32:
				ONNX_LOG("%u", *((uint32_t *)p));
				break;
			case ONNX_TENSOR_TYPE_UINT64:
				ONNX_LOG("%lu", *((uint64_t *)p));
				break;
			case ONNX_TENSOR_TYPE_BFLOAT16:
				ONNX_LOG("%g", bfloat16_to_float32(*((uint16_t *)p)));
				break;
			case ONNX_TENSOR_TYPE_FLOAT16:
				ONNX_LOG("%g", float16_to_float32(*((uint16_t *)p)));
				break;
			case ONNX_TENSOR_TYPE_FLOAT32:
				ONNX_LOG("%g", *((float *)p));
				break;
			case ONNX_TENSOR_TYPE_FLOAT64:
				ONNX_LOG("%g", *((double *)p));
				break;
			case ONNX_TENSOR_TYPE_COMPLEX64:
				ONNX_LOG("%g + %gi", *((float *)p), *((float *)(p + sizeof(float))));
				break;
			case ONNX_TENSOR_TYPE_COMPLEX128:
				ONNX_LOG("%g + %gi", *((double *)p), *((double *)(p + sizeof(double))));
				break;
			case ONNX_TENSOR_TYPE_STRING:
				ONNX_LOG("%s", (char *)(((char **)p)[0]));
				break;
			default:
				ONNX_LOG("?");
				break;
			}
			ONNX_LOG("\r\n");
		}
		else
		{
			ONNX_LOG(" = ");
			ONNX_LOG("null");
			ONNX_LOG("\r\n");
		}
	}
}

void onnx_node_dump(struct onnx_node_t * n, int detail)
{
	int i;

	if(n)
	{
		ONNX_LOG("%s: %s-%d (%s)\r\n", n->proto->name, n->proto->op_type, n->opset, (strlen(n->proto->domain) > 0) ? n->proto->domain : "ai.onnx");
		if(n->ninput > 0)
		{
			ONNX_LOG("\tInputs:\r\n");
			for(i = 0; i < n->ninput; i++)
			{
				ONNX_LOG("\t\t");
				onnx_tensor_dump(n->inputs[i], detail);
			}
		}
		if(n->noutput > 0)
		{
			ONNX_LOG("\tOutputs:\r\n");
			for(i = 0; i < n->noutput; i++)
			{
				ONNX_LOG("\t\t");
				onnx_tensor_dump(n->outputs[i], detail);
			}
		}
	}
}

void onnx_graph_dump(struct onnx_graph_t * g, int detail)
{
	int i;

	if(g)
	{
		for(i = 0; i < g->nlen; i++)
			onnx_node_dump(&g->nodes[i], detail);
	}
}

void onnx_context_dump(struct onnx_context_t * ctx, int detail)
{
	int i;

	if(ctx)
	{
		if(ctx->model)
		{
			ONNX_LOG("IR Version: v%ld\r\n", ctx->model->ir_version);
			ONNX_LOG("Producer: %s %s\r\n", ctx->model->producer_name, ctx->model->producer_version);
			ONNX_LOG("Domain: %s\r\n", ctx->model->domain);
			ONNX_LOG("Imports:\r\n");
			for(i = 0; i < ctx->model->n_opset_import; i++)
				ONNX_LOG("\t%s v%ld\r\n", (strlen(ctx->model->opset_import[i]->domain) > 0) ? ctx->model->opset_import[i]->domain : "ai.onnx", ctx->model->opset_import[i]->version);
		}
		if(ctx->g)
			onnx_graph_dump(ctx->g, detail);
	}
}

void onnx_run(struct onnx_context_t * ctx)
{
	struct onnx_node_t * n;
	int i;

	if(ctx)
	{
		for(i = 0; i < ctx->g->nlen; i++)
		{
			n = &ctx->g->nodes[i];
			if(n->reshape(n))
				n->operator(n);
		}
	}
}
