#ifndef __HELPER_H__
#define __HELPER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

static inline uint16_t float32_to_float16(float v)
{
	union { uint32_t u; float f; } t;
	uint16_t y;

	t.f = v;
	y = ((t.u & 0x7fffffff) >> 13) - (0x38000000 >> 13);
	y |= ((t.u & 0x80000000) >> 16);
	return y;
}

static inline float float16_to_float32(uint16_t v)
{
	union { uint32_t u; float f; } t;

	t.u = v;
	t.u = ((t.u & 0x7fff) << 13) + 0x38000000;
	t.u |= ((v & 0x8000) << 16);
	return t.f;
}

static inline uint16_t float32_to_bfloat16(float v)
{
	union { uint32_t u; float f; } t;

	t.f = v;
	return t.u >> 16;
}

static inline float bfloat16_to_float32(uint16_t v)
{
	union { uint32_t u; float f; } t;

	t.u = v << 16;
	return t.f;
}

#ifdef __cplusplus
}
#endif

#endif /* __HELPER_H__ */
