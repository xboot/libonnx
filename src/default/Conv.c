#include <onnx.h>

enum auto_pad_t {
	AUTO_PAD_NOTSET		= 0,
	AUTO_PAD_SAME_UPPER	= 1,
	AUTO_PAD_SAME_LOWER	= 2,
	AUTO_PAD_VALID		= 3,
};

enum conv_mode_t {
	CONV_SIMPLE = 0,
	CONV_CACHED = 1,
	CONV_IM2COL = 2,
};

struct operator_pdata_t {
	enum auto_pad_t auto_pad;
	int group;
	int * kernels;
	int nkernel;
	int * dilations;
	int ndilation;
	int * pads;
	int npad;
	int * strides;
	int nstride;

	int cpads[32];
};

static int Conv_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	int64_t * ints;
	int i, l;

	if((n->ninput >= 2) && (n->noutput == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			memset(pdat, 0, sizeof(struct operator_pdata_t));
			switch(shash(onnx_attribute_read_string(n, "auto_pad", "NOTSET")))
			{
			case 0xc3966fc2: /* "NOTSET" */
				pdat->auto_pad = AUTO_PAD_NOTSET;
				break;
			case 0xcbbc7856: /* "SAME_UPPER" */
				pdat->auto_pad = AUTO_PAD_SAME_UPPER;
				break;
			case 0xcb192d33: /* "SAME_LOWER" */
				pdat->auto_pad = AUTO_PAD_SAME_LOWER;
				break;
			case 0x0e382d15: /* "VALID" */
				pdat->auto_pad = AUTO_PAD_VALID;
				break;
			default:
				pdat->auto_pad = AUTO_PAD_NOTSET;
				break;
			}
			pdat->group = onnx_attribute_read_int(n, "group", 1);
			pdat->nkernel = onnx_attribute_read_ints(n, "kernel_shape", &ints);
			if(pdat->nkernel > 0)
			{
				pdat->kernels = malloc(sizeof(int) * pdat->nkernel);
				for(i = 0; i < pdat->nkernel; i++)
					pdat->kernels[i] = ints[i];
			}
			pdat->ndilation = pdat->nkernel;
			pdat->dilations = malloc(sizeof(int) * pdat->ndilation);
			if(pdat->dilations)
			{
				l = onnx_attribute_read_ints(n, "dilations", &ints);
				for(i = 0; i < l; i++)
					pdat->dilations[i] = ints[i];
				for(; i < pdat->ndilation; i++)
					pdat->dilations[i] = 1;
			}
			pdat->npad = pdat->nkernel * 2;
			pdat->pads = malloc(sizeof(int) * pdat->npad);
			if(pdat->pads)
			{
				l = onnx_attribute_read_ints(n, "pads", &ints);
				for(i = 0; i < l; i++)
					pdat->pads[i] = ints[i];
				for(; i < pdat->npad; i++)
					pdat->pads[i] = 0;
			}
			pdat->nstride = pdat->nkernel;
			pdat->strides = malloc(sizeof(int) * pdat->nstride);
			if(pdat->strides)
			{
				l = onnx_attribute_read_ints(n, "strides", &ints);
				for(i = 0; i < l; i++)
					pdat->strides[i] = ints[i];
				for(; i < pdat->nstride; i++)
					pdat->strides[i] = 1;
			}
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int Conv_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
	{
		if(pdat->kernels)
			free(pdat->kernels);
		if(pdat->dilations)
			free(pdat->dilations);
		if(pdat->pads)
			free(pdat->pads);
		if(pdat->strides)
			free(pdat->strides);
		free(pdat);
	}
	return 1;
}

static int Conv_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * w = n->inputs[1];
	int ndim = x->ndim;
	int dims[ndim];
	int pad;
	int i;

	switch(pdat->auto_pad)
	{
	case AUTO_PAD_NOTSET:
		memcpy(pdat->cpads, pdat->pads, sizeof(int) * pdat->npad);
		break;
	case AUTO_PAD_SAME_UPPER:
		for(i = 0; i < pdat->npad / 2; i++)
		{
			pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) - x->dims[i + 2];
			pdat->cpads[i] = pad / 2;
			pdat->cpads[i + pdat->nkernel] = pad - pdat->cpads[i];
		}
		break;
	case AUTO_PAD_SAME_LOWER:
		for(i = 0; i < pdat->npad / 2; i++)
		{
			pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) - x->dims[i + 2];
			pdat->cpads[i + pdat->nkernel] = pad / 2;
			pdat->cpads[i] = pad - pdat->cpads[i + pdat->nkernel];
		}
		break;
	case AUTO_PAD_VALID:
		memset(pdat->cpads, 0, sizeof(int) * pdat->npad);
		break;
	default:
		break;
	}
	dims[0] = x->dims[0];
	dims[1] = w->dims[0];
	for(i = 0; i < ndim - 2; i++)
	{
		switch(pdat->auto_pad)
		{
		case AUTO_PAD_NOTSET:
			dims[i + 2] = floorf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->nkernel] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1)) / (float)pdat->strides[i] + 1);
			break;
		case AUTO_PAD_SAME_UPPER:
		case AUTO_PAD_SAME_LOWER:
			dims[i + 2] = ceilf(x->dims[i + 2] / (float)pdat->strides[i]);
			break;
		case AUTO_PAD_VALID:
			dims[i + 2] = ceilf((x->dims[i + 2] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) + 1) / (float)pdat->strides[i]);
			break;
		default:
			break;
		}
	}
	return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static inline int dim_next(int ndim, int * dims, int * dim_max)
{
	if(ndim == 0)
		return 0;
	while(1)
	{
		ndim = ndim - 1;
		dims[ndim] += 1;
		if(dims[ndim] < dim_max[ndim])
			return 1;
		else
		{
			if(ndim == 0)
				return 0;
			dims[ndim] = 0;
		}
	}
}

static inline int dim_offset(int ndim, int * dims, int * dim_max)
{
	int o, s;
	int i;

	for(i = ndim - 1, o = 0, s = 1; i >= 0; i--)
	{
		o += dims[i] * s;
		s *= dim_max[i];
	}
	return o;
}

static inline void dgemm_float32(int n, int m, int o, float * A, float * B, float * C)
{
	typedef float (*atype)[o];
	typedef float (*btype)[m];
	typedef float (*ctype)[m];
	
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			((ctype)C)[i][j] = 0.;
		}
	}
	for (int i = 0; i < n; ++i)
	{
		for (int k = 0; k < o; ++k)
		{
			for (int j = 0; j < m; ++j)
			{
				((ctype)C)[i][j] += ((atype)A)[i][k] * ((btype)B)[k][j];
			}
		}
	}
}


static inline void dgemm_float64(int n, int m, int o, double * A, double * B, double * C)
{
	typedef double (*atype)[o];
	typedef double (*btype)[m];
	typedef double (*ctype)[m];
	
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			((ctype)C)[i][j] = 0.;
		}
	}
	for (int i = 0; i < n; ++i)
	{
		for (int k = 0; k < o; ++k)
		{
			for (int j = 0; j < m; ++j)
			{
				((ctype)C)[i][j] += ((atype)A)[i][k] * ((btype)B)[k][j];
			}
		}
	}
}

static void Conv_float16(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * w = n->inputs[1];
	struct onnx_tensor_t * b = NULL;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * pw = (uint16_t *)w->datas;
	uint16_t * pb = NULL;

	enum conv_mode_t conv_mode = CONV_SIMPLE;
	float * pxcache = NULL;
	float * matw = NULL;
	float * matx = NULL;
	float * maty = NULL;

	float sum, v, weight;
	int ndim = x->ndim;
	int M = w->dims[0];
	int C = w->dims[1];
	int H = w->dims[2];
	int W = w->dims[3];
	int ch, i;

	if(n->ninput > 2)
	{
		b = n->inputs[2];
		pb = (uint16_t *)b->datas;
	}
	if(ndim == 4)
	{
		int iC = x->dims[1];
		int iH = x->dims[2];
		int iW = x->dims[3];

		int oN = y->dims[0];
		int oC = w->dims[0];
		int oH = y->dims[2];
		int oW = y->dims[3];

		int MM = M / pdat->group;
		int CC = iC / pdat->group;

		typedef float (*pxtype)[iC][iH][iW];
		typedef float (*pwtype)[C][H][W];
		typedef float (*pytype)[M][oH][oW];
		typedef float (*pxcachetype)[(oC * pdat->group / M) * C][H][W];
		typedef float (*mwtype)/*[H * W * C]*/[MM];
		typedef float (*mxtype)/*[oH * oW]*/[H * W * C];
		typedef float (*mytype)/*[oH * oW]*/[MM];

		/* try im2col first */
		matw = valloc(MM * H * W * C * sizeof(float));
		matx = valloc(oH * oW * H * W * C * sizeof(float));
		maty = valloc(oH * oW * MM * sizeof(float));
		if (matw && matx && maty)
		{
			conv_mode = CONV_IM2COL;
		}
		else
		{
			if (matw) free(matw);
			if (matx) free(matx);
			if (maty) free(maty);
			
			/* then try cached conv */
			pxcache = valloc(oN * (oC * pdat->group / M) * C * H * W * sizeof(float));
			if (pxcache)
			{
				conv_mode = CONV_CACHED;
			}
		}

		if (conv_mode == CONV_SIMPLE || conv_mode == CONV_CACHED)
		{
			for(int h = 0; h < oH; ++h)
			{
				for(int w = 0; w < oW; ++w)
				{
					int base_h = h * pdat->strides[0] - pdat->cpads[0];
					int base_w = w * pdat->strides[1] - pdat->cpads[1];

					if (pxcache)
					{
						for(int n = 0; n < oN; ++n)
						{
							for(int group_c = 0; group_c < oC * pdat->group / M; ++group_c)
							{
								int base_c = group_c * C;
								for(int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i)
								{
									int input_h = base_h + i * pdat->dilations[0];
									if(input_h >= iH)
										break;
									for(int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j)
									{
										int input_w = base_w + j * pdat->dilations[1];
										if(input_w >= iW)
											break;
										for(int w_channel = 0; w_channel < C; ++w_channel)
										{
											ch = base_c + w_channel;
											((pxcachetype)pxcache)[n][ch][i][j] = float16_to_float32(((pxtype)px)[n][ch][input_h][input_w]);
										}
									}
								}
							}
						}
					}

					for(int n = 0; n < oN; ++n)
					{
						for(int c = 0; c < oC; ++c)
						{
							int base_c = (c * pdat->group / M) * C;
							sum = 0;
							for(int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i)
							{
								int input_h = base_h + i * pdat->dilations[0];
								if(input_h >= iH)
									break;
								for(int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j)
								{
									int input_w = base_w + j * pdat->dilations[1];
									if(input_w >= iW)
										break;
									for(int w_channel = 0; w_channel < C; ++w_channel)
									{
										ch = base_c + w_channel;
										if (pxcache)
										{										
											v = ((pxcachetype)pxcache)[n][ch][i][j];
										}
										else
										{
											v = float16_to_float32(((pxtype)px)[n][ch][input_h][input_w]);
										}									
										weight = float16_to_float32(((pwtype)pw)[c][w_channel][i][j]);
										sum += v * weight;
									}
								}
							}
							if(pb)
								sum += float16_to_float32(pb[c]);
							((pytype)py)[n][c][h][w] = float32_to_float16(sum);
						}
					}
				}
			}
			if (pxcache)
			{
				free(pxcache);
			}
		}
		else if (conv_mode == CONV_IM2COL)
		{			
			for (int g = 0; g < pdat->group; g++)
			{
				for (size_t m = 0; m < MM; m++)
				{
					for (size_t c = 0; c < C; c++)
					{
						for (size_t h = 0; h < H; h++)
						{
							for (size_t w = 0; w < W; w++)
							{
								((mwtype)matw)[c * H * W + h * W + w][m] = float16_to_float32(((pwtype)pw)[g * MM + m][c][h][w]);
							}						
						}					
					}				
				}
				
				for (int n = 0; n < oN; n++)
				{
					for (size_t hh = 0; hh < oH; hh++)
					{
						for (size_t ww = 0; ww < oW; ww++)
						{
							int base_h = hh * pdat->strides[0] - pdat->cpads[0];
							int base_w = ww * pdat->strides[1] - pdat->cpads[1];
							for (size_t c = 0; c < C; c++)
							{
								for (size_t h = 0; h < H; h++)
								{
									for (size_t w = 0; w < W; w++)
									{
										int ih = base_h + h * pdat->dilations[0];
										int iw = base_w + w * pdat->dilations[1];
										if (ih < 0 || iw < 0 || ih >= iH || iw >= iW)
										{
											((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = 0.;
										}
										else
										{	
											((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = float16_to_float32(((pxtype)px)[n][g * CC + c][ih][iw]);
										}
									}
								}
							}
						}
					}
					dgemm_float32(oH * oW, MM, H * W * C, matx, matw, maty);
					for (int m = 0; m < MM; ++m)
					{
						for (int h = 0; h < oH; ++h)
						{
							for (int w = 0; w < oW; ++w)
							{
								float t = ((mytype)maty)[h * oW + w][m];
								if (pb)
								{
									t += float16_to_float32(pb[g * MM + m]);
								}
								((pytype)py)[n][g * MM + m][h][w] = float32_to_float16(t);
							}
						}
					}
				}
			}
			free(matw);
			free(matx);
			free(maty);
		}
		else
		{
			/* never */
		}
	}
	else
	{
		int i_dim[ndim];
		int o_dim[ndim];
		int w_dim[ndim];
		int b_dim[ndim];

		memset(o_dim, 0, sizeof(o_dim));
		do {
			b_dim[0] = o_dim[0];
			for(i = 2; i < ndim; i++)
				b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
			sum = 0;
			memset(w_dim, 0, sizeof(w_dim));
			w_dim[0] = o_dim[1];
			do {
				if(w_dim[1] == 1)
					break;
				i_dim[0] = b_dim[0];
				for(i = 2; i < ndim; i++)
					i_dim[i] = b_dim[i] + w_dim[i] * pdat->dilations[i - 2];
				for(ch = 0; ch < C; ch++)
				{
					i_dim[1] = (o_dim[1] * pdat->group / M) * C + ch;
					w_dim[1] = ch;
					for(i = 0; i < ndim; i++)
					{
						if((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
						{
							v = 0;
							break;
						}
					}
					if(i >= ndim)
						v = float16_to_float32(px[dim_offset(ndim, i_dim, x->dims)]);
					for(i = 0; i < ndim; i++)
					{
						if((w_dim[i] < 0) || (w_dim[i] >= w->dims[i]))
						{
							weight = 0;
							break;
						}
					}
					if(i >= ndim)
						weight = float16_to_float32(pw[dim_offset(ndim, w_dim, w->dims)]);
					sum += v * weight;
				}
				w_dim[1] = 0;
			} while(dim_next(ndim, w_dim, w->dims));
			if(pb)
				sum += float16_to_float32(pb[o_dim[1]]);
			py[dim_offset(ndim, o_dim, y->dims)] = float32_to_float16(sum);
		} while(dim_next(ndim, o_dim, y->dims));
	}
}

static void Conv_float32(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * w = n->inputs[1];
	struct onnx_tensor_t * b = NULL;
	float * py = (float *)y->datas;
	float * px = (float *)x->datas;
	float * pw = (float *)w->datas;
	float * pb = NULL;

	enum conv_mode_t conv_mode = CONV_SIMPLE;
	float * pxcache = NULL;
	float * matw = NULL;
	float * matx = NULL;
	float * maty = NULL;

	float sum, v, weight;
	int ndim = x->ndim;
	int M = w->dims[0];
	int C = w->dims[1];
	int H = w->dims[2];
	int W = w->dims[3];
	int ch, i;

	if(n->ninput > 2)
	{
		b = n->inputs[2];
		pb = (float *)b->datas;
	}
	if(ndim == 4)
	{
		int iC = x->dims[1];
		int iH = x->dims[2];
		int iW = x->dims[3];

		int oN = y->dims[0];
		int oC = w->dims[0];
		int oH = y->dims[2];
		int oW = y->dims[3];

		int MM = M / pdat->group;
		int CC = iC / pdat->group;

		typedef float (*pxtype)[iC][iH][iW];
		typedef float (*pwtype)[C][H][W];
		typedef float (*pytype)[M][oH][oW];
		typedef float (*pxcachetype)[(oC * pdat->group / M) * C][H][W];
		typedef float (*mwtype)/*[H * W * C]*/[MM];
		typedef float (*mxtype)/*[oH * oW]*/[H * W * C];
		typedef float (*mytype)/*[oH * oW]*/[MM];

		/* try im2col first */
		matw = valloc(MM * H * W * C * sizeof(float));
		matx = valloc(oH * oW * H * W * C * sizeof(float));
		maty = valloc(oH * oW * MM * sizeof(float));
		if (matw && matx && maty)
		{
			conv_mode = CONV_IM2COL;
		}
		else
		{
			if (matw) free(matw);
			if (matx) free(matx);
			if (maty) free(maty);
			
			/* then try cached conv */
			pxcache = valloc(oN * (oC * pdat->group / M) * C * H * W * sizeof(float));
			if (pxcache)
			{
				conv_mode = CONV_CACHED;
			}
		}

		if (conv_mode == CONV_SIMPLE || conv_mode == CONV_CACHED)
		{
			for(int h = 0; h < oH; ++h)
			{
				for(int w = 0; w < oW; ++w)
				{
					int base_h = h * pdat->strides[0] - pdat->cpads[0];
					int base_w = w * pdat->strides[1] - pdat->cpads[1];

					if (pxcache)
					{
						for(int n = 0; n < oN; ++n)
						{
							for(int group_c = 0; group_c < oC * pdat->group / M; ++group_c)
							{
								int base_c = group_c * C;
								for(int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i)
								{
									int input_h = base_h + i * pdat->dilations[0];
									if(input_h >= iH)
										break;
									for(int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j)
									{
										int input_w = base_w + j * pdat->dilations[1];
										if(input_w >= iW)
											break;
										for(int w_channel = 0; w_channel < C; ++w_channel)
										{
											ch = base_c + w_channel;
											((pxcachetype)pxcache)[n][ch][i][j] = ((pxtype)px)[n][ch][input_h][input_w];
										}
									}
								}
							}
						}
					}

					for(int n = 0; n < oN; ++n)
					{
						for(int c = 0; c < oC; ++c)
						{
							int base_c = (c * pdat->group / M) * C;
							sum = 0;
							for(int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i)
							{
								int input_h = base_h + i * pdat->dilations[0];
								if(input_h >= iH)
									break;
								for(int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j)
								{
									int input_w = base_w + j * pdat->dilations[1];
									if(input_w >= iW)
										break;
									for(int w_channel = 0; w_channel < C; ++w_channel)
									{
										ch = base_c + w_channel;
										if (pxcache)
										{										
											v = ((pxcachetype)pxcache)[n][ch][i][j];
										}
										else
										{
											v = ((pxtype)px)[n][ch][input_h][input_w];
										}									
										weight = ((pwtype)pw)[c][w_channel][i][j];
										sum += v * weight;
									}
								}
							}
							if(pb)
								sum += pb[c];
							((pytype)py)[n][c][h][w] = sum;
						}
					}
				}
			}
			if (pxcache)
			{
				free(pxcache);
			}
		}
		else if (conv_mode == CONV_IM2COL)
		{			
			for (int g = 0; g < pdat->group; g++)
			{
				for (size_t m = 0; m < MM; m++)
				{
					for (size_t c = 0; c < C; c++)
					{
						for (size_t h = 0; h < H; h++)
						{
							for (size_t w = 0; w < W; w++)
							{
								((mwtype)matw)[c * H * W + h * W + w][m] = ((pwtype)pw)[g * MM + m][c][h][w];
							}						
						}					
					}				
				}
				
				for (int n = 0; n < oN; n++)
				{
					for (size_t hh = 0; hh < oH; hh++)
					{
						for (size_t ww = 0; ww < oW; ww++)
						{
							int base_h = hh * pdat->strides[0] - pdat->cpads[0];
							int base_w = ww * pdat->strides[1] - pdat->cpads[1];
							for (size_t c = 0; c < C; c++)
							{
								for (size_t h = 0; h < H; h++)
								{
									for (size_t w = 0; w < W; w++)
									{
										int ih = base_h + h * pdat->dilations[0];
										int iw = base_w + w * pdat->dilations[1];
										if (ih < 0 || iw < 0 || ih >= iH || iw >= iW)
										{
											((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = 0.;
										}
										else
										{	
											((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = ((pxtype)px)[n][g * CC + c][ih][iw];
										}
									}
								}
							}
						}
					}
					dgemm_float32(oH * oW, MM, H * W * C, matx, matw, maty);
					for (int m = 0; m < MM; ++m)
					{
						for (int h = 0; h < oH; ++h)
						{
							for (int w = 0; w < oW; ++w)
							{
								float t = ((mytype)maty)[h * oW + w][m];
								if (pb)
								{
									t += pb[g * MM + m];
								}
								((pytype)py)[n][g * MM + m][h][w] = t;
							}
						}
					}
				}
			}
			free(matw);
			free(matx);
			free(maty);
		}
		else
		{
			/* never */
		}
	}
	else
	{
		int i_dim[ndim];
		int o_dim[ndim];
		int w_dim[ndim];
		int b_dim[ndim];

		memset(o_dim, 0, sizeof(o_dim));
		do {
			b_dim[0] = o_dim[0];
			for(i = 2; i < ndim; i++)
				b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
			sum = 0;
			memset(w_dim, 0, sizeof(w_dim));
			w_dim[0] = o_dim[1];
			do {
				if(w_dim[1] == 1)
					break;
				i_dim[0] = b_dim[0];
				for(i = 2; i < ndim; i++)
					i_dim[i] = b_dim[i] + w_dim[i] * pdat->dilations[i - 2];
				for(ch = 0; ch < C; ch++)
				{
					i_dim[1] = (o_dim[1] * pdat->group / M) * C + ch;
					w_dim[1] = ch;
					for(i = 0; i < ndim; i++)
					{
						if((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
						{
							v = 0;
							break;
						}
					}
					if(i >= ndim)
						v = px[dim_offset(ndim, i_dim, x->dims)];
					for(i = 0; i < ndim; i++)
					{
						if((w_dim[i] < 0) || (w_dim[i] >= w->dims[i]))
						{
							weight = 0;
							break;
						}
					}
					if(i >= ndim)
						weight = pw[dim_offset(ndim, w_dim, w->dims)];
					sum += v * weight;
				}
				w_dim[1] = 0;
			} while(dim_next(ndim, w_dim, w->dims));
			if(pb)
				sum += pb[o_dim[1]];
			py[dim_offset(ndim, o_dim, y->dims)] = sum;
		} while(dim_next(ndim, o_dim, y->dims));
	}
}

static void Conv_float64(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * w = n->inputs[1];
	struct onnx_tensor_t * b = NULL;
	double * py = (double *)y->datas;
	double * px = (double *)x->datas;
	double * pw = (double *)w->datas;
	double * pb = NULL;

	enum conv_mode_t conv_mode = CONV_SIMPLE;
	double * pxcache = NULL;
	double * matw = NULL;
	double * matx = NULL;
	double * maty = NULL;

	double sum, v, weight;
	int ndim = x->ndim;
	int M = w->dims[0];
	int C = w->dims[1];
	int H = w->dims[2];
	int W = w->dims[3];
	int ch, i;

	if(n->ninput > 2)
	{
		b = n->inputs[2];
		pb = (double *)b->datas;
	}
	if(ndim == 4)
	{
		int iC = x->dims[1];
		int iH = x->dims[2];
		int iW = x->dims[3];

		int oN = y->dims[0];
		int oC = w->dims[0];
		int oH = y->dims[2];
		int oW = y->dims[3];

		int MM = M / pdat->group;
		int CC = iC / pdat->group;

		typedef double (*pxtype)[iC][iH][iW];
		typedef double (*pwtype)[C][H][W];
		typedef double (*pytype)[M][oH][oW];
		typedef double (*pxcachetype)[(oC * pdat->group / M) * C][H][W];
		typedef double (*mwtype)/*[H * W * C]*/[MM];
		typedef double (*mxtype)/*[oH * oW]*/[H * W * C];
		typedef double (*mytype)/*[oH * oW]*/[MM];

		/* try im2col first */
		matw = valloc(MM * H * W * C * sizeof(double));
		matx = valloc(oH * oW * H * W * C * sizeof(double));
		maty = valloc(oH * oW * MM * sizeof(double));
		if (matw && matx && maty)
		{
			conv_mode = CONV_IM2COL;
		}
		else
		{
			if (matw) free(matw);
			if (matx) free(matx);
			if (maty) free(maty);
			
			/* then try cached conv */
			pxcache = valloc(oN * (oC * pdat->group / M) * C * H * W * sizeof(double));
			if (pxcache)
			{
				conv_mode = CONV_CACHED;
			}
		}

		if (conv_mode == CONV_SIMPLE || conv_mode == CONV_CACHED)
		{
			for(int h = 0; h < oH; ++h)
			{
				for(int w = 0; w < oW; ++w)
				{
					int base_h = h * pdat->strides[0] - pdat->cpads[0];
					int base_w = w * pdat->strides[1] - pdat->cpads[1];

					if (pxcache)
					{
						for(int n = 0; n < oN; ++n)
						{
							for(int group_c = 0; group_c < oC * pdat->group / M; ++group_c)
							{
								int base_c = group_c * C;
								for(int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i)
								{
									int input_h = base_h + i * pdat->dilations[0];
									if(input_h >= iH)
										break;
									for(int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j)
									{
										int input_w = base_w + j * pdat->dilations[1];
										if(input_w >= iW)
											break;
										for(int w_channel = 0; w_channel < C; ++w_channel)
										{
											ch = base_c + w_channel;
											((pxcachetype)pxcache)[n][ch][i][j] = ((pxtype)px)[n][ch][input_h][input_w];
										}
									}
								}
							}
						}
					}

					for(int n = 0; n < oN; ++n)
					{
						for(int c = 0; c < oC; ++c)
						{
							int base_c = (c * pdat->group / M) * C;
							sum = 0;
							for(int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i)
							{
								int input_h = base_h + i * pdat->dilations[0];
								if(input_h >= iH)
									break;
								for(int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j)
								{
									int input_w = base_w + j * pdat->dilations[1];
									if(input_w >= iW)
										break;
									for(int w_channel = 0; w_channel < C; ++w_channel)
									{
										ch = base_c + w_channel;
										if (pxcache)
										{										
											v = ((pxcachetype)pxcache)[n][ch][i][j];
										}
										else
										{
											v = ((pxtype)px)[n][ch][input_h][input_w];
										}									
										weight = ((pwtype)pw)[c][w_channel][i][j];
										sum += v * weight;
									}
								}
							}
							if(pb)
								sum += pb[c];
							((pytype)py)[n][c][h][w] = sum;
						}
					}
				}
			}
			if (pxcache)
			{
				free(pxcache);
			}
		}
		else if (conv_mode == CONV_IM2COL)
		{			
			for (int g = 0; g < pdat->group; g++)
			{
				for (size_t m = 0; m < MM; m++)
				{
					for (size_t c = 0; c < C; c++)
					{
						for (size_t h = 0; h < H; h++)
						{
							for (size_t w = 0; w < W; w++)
							{
								((mwtype)matw)[c * H * W + h * W + w][m] = ((pwtype)pw)[g * MM + m][c][h][w];
							}						
						}					
					}				
				}
				
				for (int n = 0; n < oN; n++)
				{
					for (size_t hh = 0; hh < oH; hh++)
					{
						for (size_t ww = 0; ww < oW; ww++)
						{
							int base_h = hh * pdat->strides[0] - pdat->cpads[0];
							int base_w = ww * pdat->strides[1] - pdat->cpads[1];
							for (size_t c = 0; c < C; c++)
							{
								for (size_t h = 0; h < H; h++)
								{
									for (size_t w = 0; w < W; w++)
									{
										int ih = base_h + h * pdat->dilations[0];
										int iw = base_w + w * pdat->dilations[1];
										if (ih < 0 || iw < 0 || ih >= iH || iw >= iW)
										{
											((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = 0.;
										}
										else
										{	
											((mxtype)matx)[hh * oW + ww][c * H * W + h * W + w] = ((pxtype)px)[n][g * CC + c][ih][iw];
										}
									}
								}
							}
						}
					}
					dgemm_float64(oH * oW, MM, H * W * C, matx, matw, maty);
					for (int m = 0; m < MM; ++m)
					{
						for (int h = 0; h < oH; ++h)
						{
							for (int w = 0; w < oW; ++w)
							{
								float t = ((mytype)maty)[h * oW + w][m];
								if (pb)
								{
									t += pb[g * MM + m];
								}
								((pytype)py)[n][g * MM + m][h][w] = t;
							}
						}
					}
				}
			}
			free(matw);
			free(matx);
			free(maty);
		}
		else
		{
			/* never */
		}
	}
	else
	{
		int i_dim[ndim];
		int o_dim[ndim];
		int w_dim[ndim];
		int b_dim[ndim];

		memset(o_dim, 0, sizeof(o_dim));
		do {
			b_dim[0] = o_dim[0];
			for(i = 2; i < ndim; i++)
				b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
			sum = 0;
			memset(w_dim, 0, sizeof(w_dim));
			w_dim[0] = o_dim[1];
			do {
				if(w_dim[1] == 1)
					break;
				i_dim[0] = b_dim[0];
				for(i = 2; i < ndim; i++)
					i_dim[i] = b_dim[i] + w_dim[i] * pdat->dilations[i - 2];
				for(ch = 0; ch < C; ch++)
				{
					i_dim[1] = (o_dim[1] * pdat->group / M) * C + ch;
					w_dim[1] = ch;
					for(i = 0; i < ndim; i++)
					{
						if((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
						{
							v = 0;
							break;
						}
					}
					if(i >= ndim)
						v = px[dim_offset(ndim, i_dim, x->dims)];
					for(i = 0; i < ndim; i++)
					{
						if((w_dim[i] < 0) || (w_dim[i] >= w->dims[i]))
						{
							weight = 0;
							break;
						}
					}
					if(i >= ndim)
						weight = pw[dim_offset(ndim, w_dim, w->dims)];
					sum += v * weight;
				}
				w_dim[1] = 0;
			} while(dim_next(ndim, w_dim, w->dims));
			if(pb)
				sum += pb[o_dim[1]];
			py[dim_offset(ndim, o_dim, y->dims)] = sum;
		} while(dim_next(ndim, o_dim, y->dims));
	}
}

void resolver_default_op_Conv(struct onnx_node_t * n)
{
	if(n->opset >= 11)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Conv_init;
			n->exit = Conv_exit;
			n->reshape = Conv_reshape;
			n->operator = Conv_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Conv_init;
			n->exit = Conv_exit;
			n->reshape = Conv_reshape;
			n->operator = Conv_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Conv_init;
			n->exit = Conv_exit;
			n->reshape = Conv_reshape;
			n->operator = Conv_float64;
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
			n->init = Conv_init;
			n->exit = Conv_exit;
			n->reshape = Conv_reshape;
			n->operator = Conv_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Conv_init;
			n->exit = Conv_exit;
			n->reshape = Conv_reshape;
			n->operator = Conv_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Conv_init;
			n->exit = Conv_exit;
			n->reshape = Conv_reshape;
			n->operator = Conv_float64;
			break;
		default:
			break;
		}
	}
}
