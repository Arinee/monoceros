/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     distance.cc
 *   \author   Hechong.xyf
 *   \date     Jan 2018
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Distance (internal)
 */

#include "distance.h"
#include "cpu_features.h"
#include <algorithm>
#include <cfloat>
#include <cmath>

#define fast_abs std::abs
#define fast_sqrt std::sqrt

// #undef __AVX512F__
// #undef __AVX2__
// #undef __AVX__
// #undef __SSE4_2__
// #undef __SSE4_1__
// #undef __SSE3__
// #undef __SSE2__
// #undef __SSE__

#if defined(__SSE__)
#ifndef __FMA__
#define _mm_fmadd_ps(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
#endif

static inline float horizontal_add_v128(__m128 v)
{
#ifdef __SSE3__
    __m128 x1 = _mm_hadd_ps(v, v);
    __m128 x2 = _mm_hadd_ps(x1, x1);
    return _mm_cvtss_f32(x2);
#else
    __m128 x1 = _mm_movehl_ps(v, v);
    __m128 x2 = _mm_add_ps(v, x1);
    __m128 x3 = _mm_shuffle_ps(x2, x2, 1);
    __m128 x4 = _mm_add_ss(x2, x3);
    return _mm_cvtss_f32(x4);
#endif
}

static inline float horizontal_max_v128(__m128 v)
{
    __m128 x1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 3, 2));
    __m128 x2 = _mm_max_ps(v, x1);
    __m128 x3 = _mm_shuffle_ps(x2, x2, _MM_SHUFFLE(0, 0, 0, 1));
    __m128 x4 = _mm_max_ps(x2, x3);
    return _mm_cvtss_f32(x4);
}

static inline float horizontal_mean_v128(const float *lhs, size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);
    float sum = 0.0f;

    __m128 xmm_sum1 = _mm_setzero_ps();
    __m128 xmm_sum2 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8) {
            xmm_sum1 = _mm_add_ps(_mm_load_ps(lhs), xmm_sum1);
            xmm_sum2 = _mm_add_ps(_mm_load_ps(lhs + 4), xmm_sum2);
        }

        if ((last - last_aligned) > 3) {
            xmm_sum1 = _mm_add_ps(_mm_load_ps(lhs), xmm_sum1);
            lhs += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8) {
            xmm_sum1 = _mm_add_ps(_mm_loadu_ps(lhs), xmm_sum1);
            xmm_sum2 = _mm_add_ps(_mm_loadu_ps(lhs + 4), xmm_sum2);
        }

        if ((last - last_aligned) > 3) {
            xmm_sum1 = _mm_add_ps(_mm_loadu_ps(lhs), xmm_sum1);
            lhs += 4;
        }
    }
    sum = horizontal_add_v128(_mm_add_ps(xmm_sum1, xmm_sum2));

    switch (last - lhs) {
    case 3:
        sum += lhs[2];
        /* FALLTHRU */
    case 2:
        sum += lhs[1];
        /* FALLTHRU */
    case 1:
        sum += lhs[0];
    }
    return (sum / (float)size);
}

static inline float
squared_euclidean_distance_v128(const float *lhs, const float *rhs, size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);
    float result = 0.0f;

    __m128 xmm_sum1 = _mm_setzero_ps();
    __m128 xmm_sum2 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm0 = _mm_sub_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
            __m128 xmm1 =
                _mm_sub_ps(_mm_load_ps(lhs + 4), _mm_load_ps(rhs + 4));
            xmm_sum1 = _mm_fmadd_ps(xmm0, xmm0, xmm_sum1);
            xmm_sum2 = _mm_fmadd_ps(xmm1, xmm1, xmm_sum2);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm0 = _mm_sub_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
            xmm_sum1 = _mm_fmadd_ps(xmm0, xmm0, xmm_sum1);
            lhs += 4;
            rhs += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm0 = _mm_sub_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
            __m128 xmm1 =
                _mm_sub_ps(_mm_loadu_ps(lhs + 4), _mm_loadu_ps(rhs + 4));
            xmm_sum1 = _mm_fmadd_ps(xmm0, xmm0, xmm_sum1);
            xmm_sum2 = _mm_fmadd_ps(xmm1, xmm1, xmm_sum2);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm0 = _mm_sub_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
            xmm_sum1 = _mm_fmadd_ps(xmm0, xmm0, xmm_sum1);
            lhs += 4;
            rhs += 4;
        }
    }
    result = horizontal_add_v128(_mm_add_ps(xmm_sum1, xmm_sum2));

    float x;
    switch (last - lhs) {
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x);
    }
    return result;
}

static inline float normalized_squared_euclidean_distance_v128(const float *lhs,
                                                               const float *rhs,
                                                               size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float avg1 = horizontal_mean_v128(lhs, size);
    float avg2 = horizontal_mean_v128(rhs, size);

    __m128 xmm_sum11 = _mm_setzero_ps();
    __m128 xmm_sum12 = _mm_setzero_ps();
    __m128 xmm_sum13 = _mm_setzero_ps();
    __m128 xmm_sum21 = _mm_setzero_ps();
    __m128 xmm_sum22 = _mm_setzero_ps();
    __m128 xmm_sum23 = _mm_setzero_ps();
    __m128 xmm_avg1 = _mm_set1_ps(avg1);
    __m128 xmm_avg2 = _mm_set1_ps(avg2);

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm11 = _mm_sub_ps(_mm_load_ps(lhs), xmm_avg1);
            __m128 xmm12 = _mm_sub_ps(_mm_load_ps(rhs), xmm_avg2);
            __m128 xmm21 = _mm_sub_ps(_mm_load_ps(lhs + 4), xmm_avg1);
            __m128 xmm22 = _mm_sub_ps(_mm_load_ps(rhs + 4), xmm_avg2);
            __m128 xmm13 = _mm_sub_ps(xmm11, xmm12);
            __m128 xmm23 = _mm_sub_ps(xmm21, xmm22);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm13, xmm13, xmm_sum13);
            xmm_sum21 = _mm_fmadd_ps(xmm21, xmm21, xmm_sum21);
            xmm_sum22 = _mm_fmadd_ps(xmm22, xmm22, xmm_sum22);
            xmm_sum23 = _mm_fmadd_ps(xmm23, xmm23, xmm_sum23);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm11 = _mm_sub_ps(_mm_load_ps(lhs), xmm_avg1);
            __m128 xmm12 = _mm_sub_ps(_mm_load_ps(rhs), xmm_avg2);
            __m128 xmm13 = _mm_sub_ps(xmm11, xmm12);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm13, xmm13, xmm_sum13);
            lhs += 4;
            rhs += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm11 = _mm_sub_ps(_mm_loadu_ps(lhs), xmm_avg1);
            __m128 xmm12 = _mm_sub_ps(_mm_loadu_ps(rhs), xmm_avg2);
            __m128 xmm21 = _mm_sub_ps(_mm_loadu_ps(lhs + 4), xmm_avg1);
            __m128 xmm22 = _mm_sub_ps(_mm_loadu_ps(rhs + 4), xmm_avg2);
            __m128 xmm13 = _mm_sub_ps(xmm11, xmm12);
            __m128 xmm23 = _mm_sub_ps(xmm21, xmm22);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm13, xmm13, xmm_sum13);
            xmm_sum21 = _mm_fmadd_ps(xmm21, xmm21, xmm_sum21);
            xmm_sum22 = _mm_fmadd_ps(xmm22, xmm22, xmm_sum22);
            xmm_sum23 = _mm_fmadd_ps(xmm23, xmm23, xmm_sum23);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm11 = _mm_sub_ps(_mm_loadu_ps(lhs), xmm_avg1);
            __m128 xmm12 = _mm_sub_ps(_mm_loadu_ps(rhs), xmm_avg2);
            __m128 xmm13 = _mm_sub_ps(xmm11, xmm12);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm13, xmm13, xmm_sum13);
            lhs += 4;
            rhs += 4;
        }
    }
    sum1 = horizontal_add_v128(_mm_add_ps(xmm_sum13, xmm_sum23));
    sum2 = horizontal_add_v128(_mm_add_ps(_mm_add_ps(xmm_sum11, xmm_sum21),
                                          _mm_add_ps(xmm_sum12, xmm_sum22)));

    float x1, x2, x3;
    switch (last - lhs) {
    case 3:
        x1 = lhs[2] - avg1;
        x2 = rhs[2] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1] - avg1;
        x2 = rhs[1] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0] - avg1;
        x2 = rhs[0] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
    }
    return (sum1 / sum2 / 2.0f);
}

static inline float weighted_squared_euclidean_distance_v128(const float *lhs,
                                                             const float *rhs,
                                                             const float *wgt,
                                                             size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);
    float result = 0.0f;

    __m128 xmm_sum1 = _mm_setzero_ps();
    __m128 xmm_sum2 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0 &&
        ((uintptr_t)wgt & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8, wgt += 8) {
            __m128 xmm0 = _mm_sub_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
            __m128 xmm1 =
                _mm_sub_ps(_mm_load_ps(lhs + 4), _mm_load_ps(rhs + 4));
            xmm_sum1 = _mm_fmadd_ps(_mm_mul_ps(xmm0, xmm0), _mm_load_ps(wgt),
                                    xmm_sum1);
            xmm_sum2 = _mm_fmadd_ps(_mm_mul_ps(xmm1, xmm1),
                                    _mm_load_ps(wgt + 4), xmm_sum2);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm0 = _mm_sub_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
            xmm_sum1 = _mm_fmadd_ps(_mm_mul_ps(xmm0, xmm0), _mm_load_ps(wgt),
                                    xmm_sum1);
            lhs += 4;
            rhs += 4;
            wgt += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8, wgt += 8) {
            __m128 xmm0 = _mm_sub_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
            __m128 xmm1 =
                _mm_sub_ps(_mm_loadu_ps(lhs + 4), _mm_loadu_ps(rhs + 4));
            xmm_sum1 = _mm_fmadd_ps(_mm_mul_ps(xmm0, xmm0), _mm_loadu_ps(wgt),
                                    xmm_sum1);
            xmm_sum2 = _mm_fmadd_ps(_mm_mul_ps(xmm1, xmm1),
                                    _mm_loadu_ps(wgt + 4), xmm_sum2);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm0 = _mm_sub_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
            xmm_sum1 = _mm_fmadd_ps(_mm_mul_ps(xmm0, xmm0), _mm_loadu_ps(wgt),
                                    xmm_sum1);
            lhs += 4;
            rhs += 4;
            wgt += 4;
        }
    }
    result = horizontal_add_v128(_mm_add_ps(xmm_sum1, xmm_sum2));

    float x;
    switch (last - lhs) {
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x * wgt[2]);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x * wgt[1]);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x * wgt[0]);
    }
    return result;
}

static inline float manhattan_distance_v128(const float *lhs, const float *rhs,
                                            size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);
    float result = 0.0f;

    static const __m128 xmm_mask =
        _mm_castsi128_ps(_mm_set1_epi32(0x7fffffffu));
    __m128 xmm_sum1 = _mm_setzero_ps();
    __m128 xmm_sum2 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm0 = _mm_sub_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
            __m128 xmm1 =
                _mm_sub_ps(_mm_load_ps(lhs + 4), _mm_load_ps(rhs + 4));
            xmm_sum1 = _mm_add_ps(xmm_sum1, _mm_and_ps(xmm_mask, xmm0));
            xmm_sum2 = _mm_add_ps(xmm_sum2, _mm_and_ps(xmm_mask, xmm1));
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm0 = _mm_sub_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
            xmm_sum1 = _mm_add_ps(xmm_sum1, _mm_and_ps(xmm_mask, xmm0));
            lhs += 4;
            rhs += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm0 = _mm_sub_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
            __m128 xmm1 =
                _mm_sub_ps(_mm_loadu_ps(lhs + 4), _mm_loadu_ps(rhs + 4));
            xmm_sum1 = _mm_add_ps(xmm_sum1, _mm_and_ps(xmm_mask, xmm0));
            xmm_sum2 = _mm_add_ps(xmm_sum2, _mm_and_ps(xmm_mask, xmm1));
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm0 = _mm_sub_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
            xmm_sum1 = _mm_add_ps(xmm_sum1, _mm_and_ps(xmm_mask, xmm0));
            lhs += 4;
            rhs += 4;
        }
    }
    result = horizontal_add_v128(_mm_add_ps(xmm_sum1, xmm_sum2));

    switch (last - lhs) {
    case 3:
        result += fast_abs(lhs[2] - rhs[2]);
        /* FALLTHRU */
    case 2:
        result += fast_abs(lhs[1] - rhs[1]);
        /* FALLTHRU */
    case 1:
        result += fast_abs(lhs[0] - rhs[0]);
    }
    return result;
}

static inline float chebyshev_distance_v128(const float *lhs, const float *rhs,
                                            size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);
    float result = 0.0f;

    static const __m128 xmm_mask =
        _mm_castsi128_ps(_mm_set1_epi32(0x7fffffffu));
    __m128 xmm_max1 = _mm_setzero_ps();
    __m128 xmm_max2 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm0 = _mm_sub_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
            __m128 xmm1 =
                _mm_sub_ps(_mm_load_ps(lhs + 4), _mm_load_ps(rhs + 4));
            xmm_max1 = _mm_max_ps(xmm_max1, _mm_and_ps(xmm_mask, xmm0));
            xmm_max2 = _mm_max_ps(xmm_max2, _mm_and_ps(xmm_mask, xmm1));
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm0 = _mm_sub_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
            xmm_max1 = _mm_max_ps(xmm_max1, _mm_and_ps(xmm_mask, xmm0));
            lhs += 4;
            rhs += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm0 = _mm_sub_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
            __m128 xmm1 =
                _mm_sub_ps(_mm_loadu_ps(lhs + 4), _mm_loadu_ps(rhs + 4));
            xmm_max1 = _mm_max_ps(xmm_max1, _mm_and_ps(xmm_mask, xmm0));
            xmm_max2 = _mm_max_ps(xmm_max2, _mm_and_ps(xmm_mask, xmm1));
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm0 = _mm_sub_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
            xmm_max1 = _mm_max_ps(xmm_max1, _mm_and_ps(xmm_mask, xmm0));
            lhs += 4;
            rhs += 4;
        }
    }
    result = horizontal_max_v128(_mm_max_ps(xmm_max1, xmm_max2));

    float x;
    switch (last - lhs) {
    case 3:
        x = fast_abs(lhs[2] - rhs[2]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 2:
        x = fast_abs(lhs[1] - rhs[1]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 1:
        x = fast_abs(lhs[0] - rhs[0]);
        if (result < x) {
            result = x;
        }
    }
    return result;
}

static inline float cosine_distance_v128(const float *lhs, const float *rhs,
                                         size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    __m128 xmm_sum11 = _mm_setzero_ps();
    __m128 xmm_sum12 = _mm_setzero_ps();
    __m128 xmm_sum13 = _mm_setzero_ps();
    __m128 xmm_sum21 = _mm_setzero_ps();
    __m128 xmm_sum22 = _mm_setzero_ps();
    __m128 xmm_sum23 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm11 = _mm_load_ps(lhs);
            __m128 xmm21 = _mm_load_ps(lhs + 4);
            __m128 xmm12 = _mm_load_ps(rhs);
            __m128 xmm22 = _mm_load_ps(rhs + 4);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm12, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum13);
            xmm_sum21 = _mm_fmadd_ps(xmm21, xmm22, xmm_sum21);
            xmm_sum22 = _mm_fmadd_ps(xmm21, xmm21, xmm_sum22);
            xmm_sum23 = _mm_fmadd_ps(xmm22, xmm22, xmm_sum23);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm11 = _mm_load_ps(lhs);
            __m128 xmm12 = _mm_load_ps(rhs);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm12, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum13);
            lhs += 4;
            rhs += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm11 = _mm_loadu_ps(lhs);
            __m128 xmm21 = _mm_loadu_ps(lhs + 4);
            __m128 xmm12 = _mm_loadu_ps(rhs);
            __m128 xmm22 = _mm_loadu_ps(rhs + 4);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm12, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum13);
            xmm_sum21 = _mm_fmadd_ps(xmm21, xmm22, xmm_sum21);
            xmm_sum22 = _mm_fmadd_ps(xmm21, xmm21, xmm_sum22);
            xmm_sum23 = _mm_fmadd_ps(xmm22, xmm22, xmm_sum23);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm11 = _mm_loadu_ps(lhs);
            __m128 xmm12 = _mm_loadu_ps(rhs);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm12, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum13);
            lhs += 4;
            rhs += 4;
        }
    }
    sum1 = horizontal_add_v128(_mm_add_ps(xmm_sum11, xmm_sum21));
    sum2 = horizontal_add_v128(_mm_add_ps(xmm_sum12, xmm_sum22));
    sum3 = horizontal_add_v128(_mm_add_ps(xmm_sum13, xmm_sum23));

    float x1, x2;
    switch (last - lhs) {
    case 3:
        x1 = lhs[2];
        x2 = rhs[2];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1];
        x2 = rhs[1];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0];
        x2 = rhs[0];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
    }
    return (1.0f - sum1 / (fast_sqrt(sum2) * fast_sqrt(sum3)));
}

static inline float canberra_distance_v128(const float *lhs, const float *rhs,
                                           size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);
    float result = 0.0f;

    static const __m128 xmm_mask =
        _mm_castsi128_ps(_mm_set1_epi32(0x7fffffffu));
    __m128 xmm_sum1 = _mm_setzero_ps();
    __m128 xmm_sum2 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm10 = _mm_load_ps(lhs);
            __m128 xmm20 = _mm_load_ps(lhs + 4);
            __m128 xmm11 = _mm_load_ps(rhs);
            __m128 xmm21 = _mm_load_ps(rhs + 4);
            __m128 xmm13 = _mm_and_ps(xmm_mask, _mm_sub_ps(xmm10, xmm11));
            __m128 xmm23 = _mm_and_ps(xmm_mask, _mm_sub_ps(xmm20, xmm21));
            __m128 xmm14 = _mm_add_ps(_mm_and_ps(xmm_mask, xmm10),
                                      _mm_and_ps(xmm_mask, xmm11));
            __m128 xmm24 = _mm_add_ps(_mm_and_ps(xmm_mask, xmm20),
                                      _mm_and_ps(xmm_mask, xmm21));
            xmm_sum1 = _mm_add_ps(_mm_div_ps(xmm13, xmm14), xmm_sum1);
            xmm_sum2 = _mm_add_ps(_mm_div_ps(xmm23, xmm24), xmm_sum2);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm10 = _mm_load_ps(lhs);
            __m128 xmm11 = _mm_load_ps(rhs);
            __m128 xmm13 = _mm_and_ps(xmm_mask, _mm_sub_ps(xmm10, xmm11));
            __m128 xmm14 = _mm_add_ps(_mm_and_ps(xmm_mask, xmm10),
                                      _mm_and_ps(xmm_mask, xmm11));
            xmm_sum1 = _mm_add_ps(_mm_div_ps(xmm13, xmm14), xmm_sum1);
            lhs += 4;
            rhs += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm10 = _mm_loadu_ps(lhs);
            __m128 xmm20 = _mm_loadu_ps(lhs + 4);
            __m128 xmm11 = _mm_loadu_ps(rhs);
            __m128 xmm21 = _mm_loadu_ps(rhs + 4);
            __m128 xmm13 = _mm_and_ps(xmm_mask, _mm_sub_ps(xmm10, xmm11));
            __m128 xmm23 = _mm_and_ps(xmm_mask, _mm_sub_ps(xmm20, xmm21));
            __m128 xmm14 = _mm_add_ps(_mm_and_ps(xmm_mask, xmm10),
                                      _mm_and_ps(xmm_mask, xmm11));
            __m128 xmm24 = _mm_add_ps(_mm_and_ps(xmm_mask, xmm20),
                                      _mm_and_ps(xmm_mask, xmm21));
            xmm_sum1 = _mm_add_ps(_mm_div_ps(xmm13, xmm14), xmm_sum1);
            xmm_sum2 = _mm_add_ps(_mm_div_ps(xmm23, xmm24), xmm_sum2);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm10 = _mm_loadu_ps(lhs);
            __m128 xmm11 = _mm_loadu_ps(rhs);
            __m128 xmm13 = _mm_and_ps(xmm_mask, _mm_sub_ps(xmm10, xmm11));
            __m128 xmm14 = _mm_add_ps(_mm_and_ps(xmm_mask, xmm10),
                                      _mm_and_ps(xmm_mask, xmm11));
            xmm_sum1 = _mm_add_ps(_mm_div_ps(xmm13, xmm14), xmm_sum1);
            lhs += 4;
            rhs += 4;
        }
    }
    result = horizontal_add_v128(_mm_add_ps(xmm_sum1, xmm_sum2));

    float x1, x2;
    switch (last - lhs) {
    case 3:
        x1 = lhs[2];
        x2 = rhs[2];
        result += (fast_abs(x1 - x2) / (fast_abs(x1) + fast_abs(x2)));
        /* FALLTHRU */
    case 2:
        x1 = lhs[1];
        x2 = rhs[1];
        result += (fast_abs(x1 - x2) / (fast_abs(x1) + fast_abs(x2)));
        /* FALLTHRU */
    case 1:
        x1 = lhs[0];
        x2 = rhs[0];
        result += (fast_abs(x1 - x2) / (fast_abs(x1) + fast_abs(x2)));
    }
    return result;
}

static inline float bray_curtis_distance_v128(const float *lhs,
                                              const float *rhs, size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);
    float sum1 = 0.0f;
    float sum2 = 0.0f;

    static const __m128 xmm_mask =
        _mm_castsi128_ps(_mm_set1_epi32(0x7fffffffu));
    __m128 xmm_sum11 = _mm_setzero_ps();
    __m128 xmm_sum12 = _mm_setzero_ps();
    __m128 xmm_sum21 = _mm_setzero_ps();
    __m128 xmm_sum22 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm10 = _mm_load_ps(lhs);
            __m128 xmm20 = _mm_load_ps(lhs + 4);
            __m128 xmm11 = _mm_load_ps(rhs);
            __m128 xmm21 = _mm_load_ps(rhs + 4);
            xmm_sum11 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_sub_ps(xmm10, xmm11)), xmm_sum11);
            xmm_sum12 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_add_ps(xmm10, xmm11)), xmm_sum12);
            xmm_sum21 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_sub_ps(xmm20, xmm21)), xmm_sum21);
            xmm_sum22 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_add_ps(xmm20, xmm21)), xmm_sum22);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm10 = _mm_load_ps(lhs);
            __m128 xmm11 = _mm_load_ps(rhs);
            xmm_sum11 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_sub_ps(xmm10, xmm11)), xmm_sum11);
            xmm_sum12 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_add_ps(xmm10, xmm11)), xmm_sum12);
            lhs += 4;
            rhs += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm10 = _mm_loadu_ps(lhs);
            __m128 xmm20 = _mm_loadu_ps(lhs + 4);
            __m128 xmm11 = _mm_loadu_ps(rhs);
            __m128 xmm21 = _mm_loadu_ps(rhs + 4);
            xmm_sum11 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_sub_ps(xmm10, xmm11)), xmm_sum11);
            xmm_sum12 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_add_ps(xmm10, xmm11)), xmm_sum12);
            xmm_sum21 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_sub_ps(xmm20, xmm21)), xmm_sum21);
            xmm_sum22 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_add_ps(xmm20, xmm21)), xmm_sum22);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm10 = _mm_loadu_ps(lhs);
            __m128 xmm11 = _mm_loadu_ps(rhs);
            xmm_sum11 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_sub_ps(xmm10, xmm11)), xmm_sum11);
            xmm_sum12 = _mm_add_ps(
                _mm_and_ps(xmm_mask, _mm_add_ps(xmm10, xmm11)), xmm_sum12);
            lhs += 4;
            rhs += 4;
        }
    }
    sum1 = horizontal_add_v128(_mm_add_ps(xmm_sum11, xmm_sum21));
    sum2 = horizontal_add_v128(_mm_add_ps(xmm_sum12, xmm_sum22));

    float x1, x2;
    switch (last - lhs) {
    case 3:
        x1 = lhs[2];
        x2 = rhs[2];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1];
        x2 = rhs[1];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0];
        x2 = rhs[0];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
    }
    return (sum1 / sum2);
}

static inline float correlation_distance_v128(const float *lhs,
                                              const float *rhs, size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float avg1 = horizontal_mean_v128(lhs, size);
    float avg2 = horizontal_mean_v128(rhs, size);

    __m128 xmm_sum11 = _mm_setzero_ps();
    __m128 xmm_sum12 = _mm_setzero_ps();
    __m128 xmm_sum13 = _mm_setzero_ps();
    __m128 xmm_sum21 = _mm_setzero_ps();
    __m128 xmm_sum22 = _mm_setzero_ps();
    __m128 xmm_sum23 = _mm_setzero_ps();
    __m128 xmm_avg1 = _mm_set1_ps(avg1);
    __m128 xmm_avg2 = _mm_set1_ps(avg2);

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm11 = _mm_sub_ps(_mm_load_ps(lhs), xmm_avg1);
            __m128 xmm12 = _mm_sub_ps(_mm_load_ps(rhs), xmm_avg2);
            __m128 xmm21 = _mm_sub_ps(_mm_load_ps(lhs + 4), xmm_avg1);
            __m128 xmm22 = _mm_sub_ps(_mm_load_ps(rhs + 4), xmm_avg2);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm12, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum13);
            xmm_sum21 = _mm_fmadd_ps(xmm21, xmm22, xmm_sum21);
            xmm_sum22 = _mm_fmadd_ps(xmm21, xmm21, xmm_sum22);
            xmm_sum23 = _mm_fmadd_ps(xmm22, xmm22, xmm_sum23);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm11 = _mm_sub_ps(_mm_load_ps(lhs), xmm_avg1);
            __m128 xmm12 = _mm_sub_ps(_mm_load_ps(rhs), xmm_avg2);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm12, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum13);
            lhs += 4;
            rhs += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm11 = _mm_sub_ps(_mm_loadu_ps(lhs), xmm_avg1);
            __m128 xmm12 = _mm_sub_ps(_mm_loadu_ps(rhs), xmm_avg2);
            __m128 xmm21 = _mm_sub_ps(_mm_loadu_ps(lhs + 4), xmm_avg1);
            __m128 xmm22 = _mm_sub_ps(_mm_loadu_ps(rhs + 4), xmm_avg2);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm12, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum13);
            xmm_sum21 = _mm_fmadd_ps(xmm21, xmm22, xmm_sum21);
            xmm_sum22 = _mm_fmadd_ps(xmm21, xmm21, xmm_sum22);
            xmm_sum23 = _mm_fmadd_ps(xmm22, xmm22, xmm_sum23);
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm11 = _mm_sub_ps(_mm_loadu_ps(lhs), xmm_avg1);
            __m128 xmm12 = _mm_sub_ps(_mm_loadu_ps(rhs), xmm_avg2);
            xmm_sum11 = _mm_fmadd_ps(xmm11, xmm12, xmm_sum11);
            xmm_sum12 = _mm_fmadd_ps(xmm11, xmm11, xmm_sum12);
            xmm_sum13 = _mm_fmadd_ps(xmm12, xmm12, xmm_sum13);
            lhs += 4;
            rhs += 4;
        }
    }
    sum1 = horizontal_add_v128(_mm_add_ps(xmm_sum11, xmm_sum21));
    sum2 = horizontal_add_v128(_mm_add_ps(xmm_sum12, xmm_sum22));
    sum3 = horizontal_add_v128(_mm_add_ps(xmm_sum13, xmm_sum23));

    float x1, x2;
    switch (last - lhs) {
    case 3:
        x1 = lhs[2] - avg1;
        x2 = rhs[2] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1] - avg1;
        x2 = rhs[1] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0] - avg1;
        x2 = rhs[0] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
    }
    return (1.0f - sum1 / (fast_sqrt(sum2) * fast_sqrt(sum3)));
}

static inline float binary_distance_v128(const float *lhs, const float *rhs,
                                         size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm0 = _mm_cmpeq_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
            __m128 xmm1 =
                _mm_cmpeq_ps(_mm_load_ps(lhs + 4), _mm_load_ps(rhs + 4));

            if (_mm_movemask_ps(xmm0) != 0xf || _mm_movemask_ps(xmm1) != 0xf) {
                return 1.0f;
            }
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm0 = _mm_cmpeq_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
            if (_mm_movemask_ps(xmm0) != 0xf) {
                return 1.0f;
            }
            lhs += 4;
            rhs += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128 xmm0 = _mm_cmpeq_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
            __m128 xmm1 =
                _mm_cmpeq_ps(_mm_loadu_ps(lhs + 4), _mm_loadu_ps(rhs + 4));

            if (_mm_movemask_ps(xmm0) != 0xf || _mm_movemask_ps(xmm1) != 0xf) {
                return 1.0f;
            }
        }

        if ((last - last_aligned) > 3) {
            __m128 xmm0 = _mm_cmpeq_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
            if (_mm_movemask_ps(xmm0) != 0xf) {
                return 1.0f;
            }
            lhs += 4;
            rhs += 4;
        }
    }

    switch (last - lhs) {
    case 3:
        if (fast_abs(lhs[2] - rhs[2]) > FLT_EPSILON) {
            return 1.0f;
        }
        /* FALLTHRU */
    case 2:
        if (fast_abs(lhs[1] - rhs[1]) > FLT_EPSILON) {
            return 1.0f;
        }
        /* FALLTHRU */
    case 1:
        if (fast_abs(lhs[0] - rhs[0]) > FLT_EPSILON) {
            return 1.0f;
        }
    }
    return 0.0f;
}

static inline float inner_product_v128(const float *lhs, const float *rhs,
                                       size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 3) << 3);
    float sum = 0.0f;

    __m128 xmm_sum1 = _mm_setzero_ps();
    __m128 xmm_sum2 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            xmm_sum1 =
                _mm_fmadd_ps(_mm_load_ps(lhs), _mm_load_ps(rhs), xmm_sum1);
            xmm_sum2 = _mm_fmadd_ps(_mm_load_ps(lhs + 4), _mm_load_ps(rhs + 4),
                                    xmm_sum2);
        }

        if ((last - last_aligned) > 3) {
            xmm_sum1 =
                _mm_fmadd_ps(_mm_load_ps(lhs), _mm_load_ps(rhs), xmm_sum1);
            lhs += 4;
            rhs += 4;
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            xmm_sum1 =
                _mm_fmadd_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs), xmm_sum1);
            xmm_sum2 = _mm_fmadd_ps(_mm_loadu_ps(lhs + 4),
                                    _mm_loadu_ps(rhs + 4), xmm_sum2);
        }

        if ((last - last_aligned) > 3) {
            xmm_sum1 =
                _mm_fmadd_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs), xmm_sum1);
            lhs += 4;
            rhs += 4;
        }
    }
    sum = horizontal_add_v128(_mm_add_ps(xmm_sum1, xmm_sum2));

    switch (last - lhs) {
    case 3:
        sum += (lhs[2] * rhs[2]);
        /* FALLTHRU */
    case 2:
        sum += (lhs[1] * rhs[1]);
        /* FALLTHRU */
    case 1:
        sum += (lhs[0] * rhs[0]);
    }
    return sum;
}
#endif // __SSE__

#if defined(__SSE2__)
#ifndef __SSE3__
#define _mm_lddqu_si128 _mm_loadu_si128
#endif // !__SSE3__

static inline int32_t horizontal_add_v128(__m128i v)
{
#ifdef __SSE3__
    __m128i x1 = _mm_hadd_epi32(v, v);
    __m128i x2 = _mm_hadd_epi32(x1, x1);
    return _mm_cvtsi128_si32(x2);
#else
    __m128i x1 = _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 0, 3, 2));
    __m128i x2 = _mm_add_epi32(v, x1);
    __m128i x3 = _mm_shuffle_epi32(x2, _MM_SHUFFLE(0, 0, 0, 1));
    __m128i x4 = _mm_add_epi32(x2, x3);
    return _mm_cvtsi128_si32(x4);
#endif
}

static inline int32_t horizontal_max_v128(__m128i v)
{
#ifdef __SSE4_1__
    __m128i x1 = _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 0, 3, 2));
    __m128i x2 = _mm_max_epi32(v, x1);
    __m128i x3 = _mm_shuffle_epi32(x2, _MM_SHUFFLE(0, 0, 0, 1));
    __m128i x4 = _mm_max_epi32(x2, x3);
    return _mm_cvtsi128_si32(x4);
#else
    return horizontal_max_v128(_mm_cvtepi32_ps(v));
#endif
}

static inline float squared_euclidean_distance_v128(const int16_t *lhs,
                                                    const int16_t *rhs,
                                                    size_t size)
{
    const int16_t *last = lhs + size;
    const int16_t *last_aligned = lhs + ((size >> 3) << 3);
    float result = 0.0f;

    static const __m128i ixmm_zero = _mm_setzero_si128();
    __m128 xmm_sum1 = _mm_setzero_ps();
    __m128 xmm_sum2 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));

            __m128 xmm0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(ixmm2, ixmm_zero));
            __m128 xmm1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(ixmm2, ixmm_zero));
            xmm_sum1 = _mm_fmadd_ps(xmm0, xmm0, xmm_sum1);
            xmm_sum2 = _mm_fmadd_ps(xmm1, xmm1, xmm_sum2);
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));

            __m128 xmm0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(ixmm2, ixmm_zero));
            __m128 xmm1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(ixmm2, ixmm_zero));
            xmm_sum1 = _mm_fmadd_ps(xmm0, xmm0, xmm_sum1);
            xmm_sum2 = _mm_fmadd_ps(xmm1, xmm1, xmm_sum2);
        }
    }
    result = horizontal_add_v128(_mm_add_ps(xmm_sum1, xmm_sum2));

    float x;
    switch (last - last_aligned) {
    case 7:
        x = lhs[6] - rhs[6];
        result += (x * x);
        /* FALLTHRU */
    case 6:
        x = lhs[5] - rhs[5];
        result += (x * x);
        /* FALLTHRU */
    case 5:
        x = lhs[4] - rhs[4];
        result += (x * x);
        /* FALLTHRU */
    case 4:
        x = lhs[3] - rhs[3];
        result += (x * x);
        /* FALLTHRU */
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x);
    }
    return result;
}

static inline float weighted_squared_euclidean_distance_v128(const int16_t *lhs,
                                                             const int16_t *rhs,
                                                             const float *wgt,
                                                             size_t size)
{
    const int16_t *last = lhs + size;
    const int16_t *last_aligned = lhs + ((size >> 3) << 3);
    float result = 0.0f;

    static const __m128i ixmm_zero = _mm_setzero_si128();
    __m128 xmm_sum1 = _mm_setzero_ps();
    __m128 xmm_sum2 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0 &&
        ((uintptr_t)wgt & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8, wgt += 8) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));

            __m128 xmm0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(ixmm2, ixmm_zero));
            __m128 xmm1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(ixmm2, ixmm_zero));
            xmm_sum1 = _mm_fmadd_ps(_mm_mul_ps(xmm0, xmm0), _mm_load_ps(wgt),
                                    xmm_sum1);
            xmm_sum2 = _mm_fmadd_ps(_mm_mul_ps(xmm1, xmm1),
                                    _mm_load_ps(wgt + 4), xmm_sum2);
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8, wgt += 8) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));

            __m128 xmm0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(ixmm2, ixmm_zero));
            __m128 xmm1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(ixmm2, ixmm_zero));
            xmm_sum1 = _mm_fmadd_ps(_mm_mul_ps(xmm0, xmm0), _mm_loadu_ps(wgt),
                                    xmm_sum1);
            xmm_sum2 = _mm_fmadd_ps(_mm_mul_ps(xmm1, xmm1),
                                    _mm_loadu_ps(wgt + 4), xmm_sum2);
        }
    }
    result = horizontal_add_v128(_mm_add_ps(xmm_sum1, xmm_sum2));

    float x;
    switch (last - last_aligned) {
    case 7:
        x = lhs[6] - rhs[6];
        result += (x * x * wgt[6]);
        /* FALLTHRU */
    case 6:
        x = lhs[5] - rhs[5];
        result += (x * x * wgt[5]);
        /* FALLTHRU */
    case 5:
        x = lhs[4] - rhs[4];
        result += (x * x * wgt[4]);
        /* FALLTHRU */
    case 4:
        x = lhs[3] - rhs[3];
        result += (x * x * wgt[3]);
        /* FALLTHRU */
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x * wgt[2]);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x * wgt[1]);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x * wgt[0]);
    }
    return result;
}

static inline float manhattan_distance_v128(const int16_t *lhs,
                                            const int16_t *rhs, size_t size)
{
    const int16_t *last = lhs + size;
    const int16_t *last_aligned = lhs + ((size >> 4) << 4);
    float result = 0.0f;

    static const __m128i ixmm_zero = _mm_setzero_si128();
    __m128i ixmm_sum1 = _mm_setzero_si128();
    __m128i ixmm_sum2 = _mm_setzero_si128();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));

            __m128i ixmm3 = _mm_load_si128((const __m128i *)(lhs + 8));
            __m128i ixmm4 = _mm_load_si128((const __m128i *)(rhs + 8));
            __m128i ixmm5 = _mm_sub_epi16(_mm_max_epi16(ixmm3, ixmm4),
                                          _mm_min_epi16(ixmm3, ixmm4));

            ixmm_sum1 = _mm_add_epi32(
                _mm_add_epi32(_mm_unpacklo_epi16(ixmm2, ixmm_zero),
                              _mm_unpackhi_epi16(ixmm2, ixmm_zero)),
                ixmm_sum1);
            ixmm_sum2 = _mm_add_epi32(
                _mm_add_epi32(_mm_unpacklo_epi16(ixmm5, ixmm_zero),
                              _mm_unpackhi_epi16(ixmm5, ixmm_zero)),
                ixmm_sum2);
        }

        if ((last - last_aligned) > 7) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));
            ixmm_sum1 = _mm_add_epi32(
                _mm_add_epi32(_mm_unpacklo_epi16(ixmm2, ixmm_zero),
                              _mm_unpackhi_epi16(ixmm2, ixmm_zero)),
                ixmm_sum1);
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));

            __m128i ixmm3 = _mm_lddqu_si128((const __m128i *)(lhs + 8));
            __m128i ixmm4 = _mm_lddqu_si128((const __m128i *)(rhs + 8));
            __m128i ixmm5 = _mm_sub_epi16(_mm_max_epi16(ixmm3, ixmm4),
                                          _mm_min_epi16(ixmm3, ixmm4));

            ixmm_sum1 = _mm_add_epi32(
                _mm_add_epi32(_mm_unpacklo_epi16(ixmm2, ixmm_zero),
                              _mm_unpackhi_epi16(ixmm2, ixmm_zero)),
                ixmm_sum1);
            ixmm_sum2 = _mm_add_epi32(
                _mm_add_epi32(_mm_unpacklo_epi16(ixmm5, ixmm_zero),
                              _mm_unpackhi_epi16(ixmm5, ixmm_zero)),
                ixmm_sum2);
        }

        if ((last - last_aligned) > 7) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));
            ixmm_sum1 = _mm_add_epi32(
                _mm_add_epi32(_mm_unpacklo_epi16(ixmm2, ixmm_zero),
                              _mm_unpackhi_epi16(ixmm2, ixmm_zero)),
                ixmm_sum1);
            lhs += 8;
            rhs += 8;
        }
    }
    result = horizontal_add_v128(
        _mm_cvtepi32_ps(_mm_add_epi32(ixmm_sum1, ixmm_sum2)));

    switch (last - lhs) {
    case 7:
        result += fast_abs(lhs[6] - rhs[6]);
        /* FALLTHRU */
    case 6:
        result += fast_abs(lhs[5] - rhs[5]);
        /* FALLTHRU */
    case 5:
        result += fast_abs(lhs[4] - rhs[4]);
        /* FALLTHRU */
    case 4:
        result += fast_abs(lhs[3] - rhs[3]);
        /* FALLTHRU */
    case 3:
        result += fast_abs(lhs[2] - rhs[2]);
        /* FALLTHRU */
    case 2:
        result += fast_abs(lhs[1] - rhs[1]);
        /* FALLTHRU */
    case 1:
        result += fast_abs(lhs[0] - rhs[0]);
    }
    return result;
}

static inline float cosine_distance_v128(const int16_t *lhs, const int16_t *rhs,
                                         size_t size)
{
    const int16_t *last = lhs + size;
    const int16_t *last_aligned = lhs + ((size >> 4) << 4);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    __m128 xmm_sum1 = _mm_setzero_ps();
    __m128 xmm_sum2 = _mm_setzero_ps();
    __m128 xmm_sum3 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_madd_epi16(ixmm0, ixmm0);
            __m128i ixmm3 = _mm_madd_epi16(ixmm1, ixmm1);
            __m128i ixmm4 = _mm_madd_epi16(ixmm0, ixmm1);

            xmm_sum2 = _mm_add_ps(_mm_cvtepi32_ps(ixmm2), xmm_sum2);
            xmm_sum3 = _mm_add_ps(_mm_cvtepi32_ps(ixmm3), xmm_sum3);
            xmm_sum1 = _mm_add_ps(_mm_cvtepi32_ps(ixmm4), xmm_sum1);

            ixmm0 = _mm_load_si128((const __m128i *)(lhs + 8));
            ixmm1 = _mm_load_si128((const __m128i *)(rhs + 8));
            ixmm2 = _mm_madd_epi16(ixmm0, ixmm0);
            ixmm3 = _mm_madd_epi16(ixmm1, ixmm1);
            ixmm4 = _mm_madd_epi16(ixmm0, ixmm1);

            xmm_sum2 = _mm_add_ps(_mm_cvtepi32_ps(ixmm2), xmm_sum2);
            xmm_sum3 = _mm_add_ps(_mm_cvtepi32_ps(ixmm3), xmm_sum3);
            xmm_sum1 = _mm_add_ps(_mm_cvtepi32_ps(ixmm4), xmm_sum1);
        }

        if ((last - last_aligned) > 7) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_madd_epi16(ixmm0, ixmm0);
            __m128i ixmm3 = _mm_madd_epi16(ixmm1, ixmm1);
            __m128i ixmm4 = _mm_madd_epi16(ixmm0, ixmm1);

            xmm_sum2 = _mm_add_ps(_mm_cvtepi32_ps(ixmm2), xmm_sum2);
            xmm_sum3 = _mm_add_ps(_mm_cvtepi32_ps(ixmm3), xmm_sum3);
            xmm_sum1 = _mm_add_ps(_mm_cvtepi32_ps(ixmm4), xmm_sum1);
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_madd_epi16(ixmm0, ixmm0);
            __m128i ixmm3 = _mm_madd_epi16(ixmm1, ixmm1);
            __m128i ixmm4 = _mm_madd_epi16(ixmm0, ixmm1);

            xmm_sum2 = _mm_add_ps(_mm_cvtepi32_ps(ixmm2), xmm_sum2);
            xmm_sum3 = _mm_add_ps(_mm_cvtepi32_ps(ixmm3), xmm_sum3);
            xmm_sum1 = _mm_add_ps(_mm_cvtepi32_ps(ixmm4), xmm_sum1);

            ixmm0 = _mm_lddqu_si128((const __m128i *)(lhs + 8));
            ixmm1 = _mm_lddqu_si128((const __m128i *)(rhs + 8));
            ixmm2 = _mm_madd_epi16(ixmm0, ixmm0);
            ixmm3 = _mm_madd_epi16(ixmm1, ixmm1);
            ixmm4 = _mm_madd_epi16(ixmm0, ixmm1);

            xmm_sum2 = _mm_add_ps(_mm_cvtepi32_ps(ixmm2), xmm_sum2);
            xmm_sum3 = _mm_add_ps(_mm_cvtepi32_ps(ixmm3), xmm_sum3);
            xmm_sum1 = _mm_add_ps(_mm_cvtepi32_ps(ixmm4), xmm_sum1);
        }

        if ((last - last_aligned) > 7) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_madd_epi16(ixmm0, ixmm0);
            __m128i ixmm3 = _mm_madd_epi16(ixmm1, ixmm1);
            __m128i ixmm4 = _mm_madd_epi16(ixmm0, ixmm1);

            xmm_sum2 = _mm_add_ps(_mm_cvtepi32_ps(ixmm2), xmm_sum2);
            xmm_sum3 = _mm_add_ps(_mm_cvtepi32_ps(ixmm3), xmm_sum3);
            xmm_sum1 = _mm_add_ps(_mm_cvtepi32_ps(ixmm4), xmm_sum1);
            lhs += 8;
            rhs += 8;
        }
    }
    sum1 = horizontal_add_v128(xmm_sum1);
    sum2 = horizontal_add_v128(xmm_sum2);
    sum3 = horizontal_add_v128(xmm_sum3);

    float x1, x2;
    switch (last - lhs) {
    case 7:
        x1 = lhs[6];
        x2 = rhs[6];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 6:
        x1 = lhs[5];
        x2 = rhs[5];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 5:
        x1 = lhs[4];
        x2 = rhs[4];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 4:
        x1 = lhs[3];
        x2 = rhs[3];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 3:
        x1 = lhs[2];
        x2 = rhs[2];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1];
        x2 = rhs[1];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0];
        x2 = rhs[0];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
    }
    return (1.0f - sum1 / (fast_sqrt(sum2) * fast_sqrt(sum3)));
}

static inline float inner_product_v128(const int16_t *lhs, const int16_t *rhs,
                                       size_t size)
{
    const int16_t *last = lhs + size;
    const int16_t *last_aligned = lhs + ((size >> 4) << 4);
    float sum = 0.0f;

    __m128 xmm_sum1 = _mm_setzero_ps();
    __m128 xmm_sum2 = _mm_setzero_ps();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m128i ixmm0 =
                _mm_madd_epi16(_mm_load_si128((const __m128i *)lhs),
                               _mm_load_si128((const __m128i *)rhs));
            __m128i ixmm1 =
                _mm_madd_epi16(_mm_load_si128((const __m128i *)(lhs + 8)),
                               _mm_load_si128((const __m128i *)(rhs + 8)));

            xmm_sum1 = _mm_add_ps(_mm_cvtepi32_ps(ixmm0), xmm_sum1);
            xmm_sum2 = _mm_add_ps(_mm_cvtepi32_ps(ixmm1), xmm_sum2);
        }

        if ((last - last_aligned) > 7) {
            xmm_sum1 = _mm_add_ps(_mm_cvtepi32_ps(_mm_madd_epi16(
                                      _mm_load_si128((const __m128i *)lhs),
                                      _mm_load_si128((const __m128i *)rhs))),
                                  xmm_sum1);
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m128i ixmm0 =
                _mm_madd_epi16(_mm_lddqu_si128((const __m128i *)lhs),
                               _mm_lddqu_si128((const __m128i *)rhs));
            __m128i ixmm1 =
                _mm_madd_epi16(_mm_lddqu_si128((const __m128i *)(lhs + 8)),
                               _mm_lddqu_si128((const __m128i *)(rhs + 8)));

            xmm_sum1 = _mm_add_ps(_mm_cvtepi32_ps(ixmm0), xmm_sum1);
            xmm_sum2 = _mm_add_ps(_mm_cvtepi32_ps(ixmm1), xmm_sum2);
        }

        if ((last - last_aligned) > 7) {
            xmm_sum1 = _mm_add_ps(_mm_cvtepi32_ps(_mm_madd_epi16(
                                      _mm_lddqu_si128((const __m128i *)lhs),
                                      _mm_lddqu_si128((const __m128i *)rhs))),
                                  xmm_sum1);
            lhs += 8;
            rhs += 8;
        }
    }
    sum = horizontal_add_v128(_mm_add_ps(xmm_sum1, xmm_sum2));

    switch (last - lhs) {
    case 7:
        sum += (lhs[6] * rhs[6]);
        /* FALLTHRU */
    case 6:
        sum += (lhs[5] * rhs[5]);
        /* FALLTHRU */
    case 5:
        sum += (lhs[4] * rhs[4]);
        /* FALLTHRU */
    case 4:
        sum += (lhs[3] * rhs[3]);
        /* FALLTHRU */
    case 3:
        sum += (lhs[2] * rhs[2]);
        /* FALLTHRU */
    case 2:
        sum += (lhs[1] * rhs[1]);
        /* FALLTHRU */
    case 1:
        sum += (lhs[0] * rhs[0]);
    }
    return sum;
}
#endif // __SSE2__

#if defined(__SSE4_1__)
static inline float chebyshev_distance_v128(const int16_t *lhs,
                                            const int16_t *rhs, size_t size)
{
    const int16_t *last = lhs + size;
    const int16_t *last_aligned = lhs + ((size >> 4) << 4);
    int32_t result = 0;

    static const __m128i ixmm_zero = _mm_setzero_si128();
    __m128i ixmm_max = _mm_setzero_si128();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));

            __m128i ixmm3 = _mm_load_si128((const __m128i *)(lhs + 8));
            __m128i ixmm4 = _mm_load_si128((const __m128i *)(rhs + 8));
            __m128i ixmm5 = _mm_sub_epi16(_mm_max_epi16(ixmm3, ixmm4),
                                          _mm_min_epi16(ixmm3, ixmm4));

            ixmm_max = _mm_max_epu16(_mm_max_epu16(ixmm2, ixmm5), ixmm_max);
        }

        if ((last - last_aligned) > 7) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));
            ixmm_max = _mm_max_epu16(ixmm2, ixmm_max);
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));

            __m128i ixmm3 = _mm_lddqu_si128((const __m128i *)(lhs + 8));
            __m128i ixmm4 = _mm_lddqu_si128((const __m128i *)(rhs + 8));
            __m128i ixmm5 = _mm_sub_epi16(_mm_max_epi16(ixmm3, ixmm4),
                                          _mm_min_epi16(ixmm3, ixmm4));

            ixmm_max = _mm_max_epu16(_mm_max_epu16(ixmm2, ixmm5), ixmm_max);
        }

        if ((last - last_aligned) > 7) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi16(_mm_max_epi16(ixmm0, ixmm1),
                                          _mm_min_epi16(ixmm0, ixmm1));
            ixmm_max = _mm_max_epu16(ixmm2, ixmm_max);
            lhs += 8;
            rhs += 8;
        }
    }
    result = horizontal_max_v128(
        _mm_max_epi32(_mm_unpacklo_epi16(ixmm_max, ixmm_zero),
                      _mm_unpackhi_epi16(ixmm_max, ixmm_zero)));

    int32_t x;
    switch (last - lhs) {
    case 7:
        x = fast_abs(lhs[6] - rhs[6]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 6:
        x = fast_abs(lhs[5] - rhs[5]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 5:
        x = fast_abs(lhs[4] - rhs[4]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 4:
        x = fast_abs(lhs[3] - rhs[3]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 3:
        x = fast_abs(lhs[2] - rhs[2]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 2:
        x = fast_abs(lhs[1] - rhs[1]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 1:
        x = fast_abs(lhs[0] - rhs[0]);
        if (result < x) {
            result = x;
        }
    }
    return result;
}

static inline float squared_euclidean_distance_v128(const int8_t *lhs,
                                                    const int8_t *rhs,
                                                    size_t size)
{
    const int8_t *last = lhs + size;
    const int8_t *last_aligned = lhs + ((size >> 5) << 5);
    float result = 0.0;

    static const __m128i ixmm_zero = _mm_setzero_si128();
    __m128i ixmm_sum1 = _mm_setzero_si128();
    __m128i ixmm_sum2 = _mm_setzero_si128();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));

            __m128i ixmm3 = _mm_load_si128((const __m128i *)(lhs + 16));
            __m128i ixmm4 = _mm_load_si128((const __m128i *)(rhs + 16));
            __m128i ixmm5 = _mm_sub_epi8(_mm_max_epi8(ixmm3, ixmm4),
                                         _mm_min_epi8(ixmm3, ixmm4));

            ixmm0 = _mm_unpacklo_epi8(ixmm2, ixmm_zero);
            ixmm1 = _mm_unpackhi_epi8(ixmm2, ixmm_zero);
            ixmm3 = _mm_unpacklo_epi8(ixmm5, ixmm_zero);
            ixmm4 = _mm_unpackhi_epi8(ixmm5, ixmm_zero);
            ixmm2 = _mm_add_epi32(_mm_madd_epi16(ixmm0, ixmm0),
                                  _mm_madd_epi16(ixmm1, ixmm1));
            ixmm5 = _mm_add_epi32(_mm_madd_epi16(ixmm3, ixmm3),
                                  _mm_madd_epi16(ixmm4, ixmm4));

            ixmm_sum1 = _mm_add_epi32(ixmm2, ixmm_sum1);
            ixmm_sum2 = _mm_add_epi32(ixmm5, ixmm_sum2);
        }

        if ((last - last_aligned) > 15) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));
            ixmm0 = _mm_unpacklo_epi8(ixmm2, ixmm_zero);
            ixmm1 = _mm_unpackhi_epi8(ixmm2, ixmm_zero);
            ixmm2 = _mm_add_epi32(_mm_madd_epi16(ixmm0, ixmm0),
                                  _mm_madd_epi16(ixmm1, ixmm1));

            ixmm_sum1 = _mm_add_epi32(ixmm2, ixmm_sum1);
            lhs += 16;
            rhs += 16;
        }
    } else {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));

            __m128i ixmm3 = _mm_lddqu_si128((const __m128i *)(lhs + 16));
            __m128i ixmm4 = _mm_lddqu_si128((const __m128i *)(rhs + 16));
            __m128i ixmm5 = _mm_sub_epi8(_mm_max_epi8(ixmm3, ixmm4),
                                         _mm_min_epi8(ixmm3, ixmm4));

            ixmm0 = _mm_unpacklo_epi8(ixmm2, ixmm_zero);
            ixmm1 = _mm_unpackhi_epi8(ixmm2, ixmm_zero);
            ixmm3 = _mm_unpacklo_epi8(ixmm5, ixmm_zero);
            ixmm4 = _mm_unpackhi_epi8(ixmm5, ixmm_zero);
            ixmm2 = _mm_add_epi32(_mm_madd_epi16(ixmm0, ixmm0),
                                  _mm_madd_epi16(ixmm1, ixmm1));
            ixmm5 = _mm_add_epi32(_mm_madd_epi16(ixmm3, ixmm3),
                                  _mm_madd_epi16(ixmm4, ixmm4));

            ixmm_sum1 = _mm_add_epi32(ixmm2, ixmm_sum1);
            ixmm_sum2 = _mm_add_epi32(ixmm5, ixmm_sum2);
        }

        if ((last - last_aligned) > 15) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));
            ixmm0 = _mm_unpacklo_epi8(ixmm2, ixmm_zero);
            ixmm1 = _mm_unpackhi_epi8(ixmm2, ixmm_zero);
            ixmm2 = _mm_add_epi32(_mm_madd_epi16(ixmm0, ixmm0),
                                  _mm_madd_epi16(ixmm1, ixmm1));

            ixmm_sum1 = _mm_add_epi32(ixmm2, ixmm_sum1);
            lhs += 16;
            rhs += 16;
        }
    }
    result = horizontal_add_v128(
        _mm_cvtepi32_ps(_mm_add_epi32(ixmm_sum1, ixmm_sum2)));

    int32_t x;
    switch (last - lhs) {
    case 15:
        x = lhs[14] - rhs[14];
        result += (x * x);
        /* FALLTHRU */
    case 14:
        x = lhs[13] - rhs[13];
        result += (x * x);
        /* FALLTHRU */
    case 13:
        x = lhs[12] - rhs[12];
        result += (x * x);
        /* FALLTHRU */
    case 12:
        x = lhs[11] - rhs[11];
        result += (x * x);
        /* FALLTHRU */
    case 11:
        x = lhs[10] - rhs[10];
        result += (x * x);
        /* FALLTHRU */
    case 10:
        x = lhs[9] - rhs[9];
        result += (x * x);
        /* FALLTHRU */
    case 9:
        x = lhs[8] - rhs[8];
        result += (x * x);
        /* FALLTHRU */
    case 8:
        x = lhs[7] - rhs[7];
        result += (x * x);
        /* FALLTHRU */
    case 7:
        x = lhs[6] - rhs[6];
        result += (x * x);
        /* FALLTHRU */
    case 6:
        x = lhs[5] - rhs[5];
        result += (x * x);
        /* FALLTHRU */
    case 5:
        x = lhs[4] - rhs[4];
        result += (x * x);
        /* FALLTHRU */
    case 4:
        x = lhs[3] - rhs[3];
        result += (x * x);
        /* FALLTHRU */
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x);
    }
    return result;
}

static inline float manhattan_distance_v128(const int8_t *lhs,
                                            const int8_t *rhs, size_t size)
{
    const int8_t *last = lhs + size;
    const int8_t *last_aligned = lhs + ((size >> 5) << 5);
    int32_t result = 0;

    static const __m128i ixmm_zero = _mm_setzero_si128();
    __m128i ixmm_sum1 = _mm_setzero_si128();
    __m128i ixmm_sum2 = _mm_setzero_si128();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));

            __m128i ixmm3 = _mm_load_si128((const __m128i *)(lhs + 16));
            __m128i ixmm4 = _mm_load_si128((const __m128i *)(rhs + 16));
            __m128i ixmm5 = _mm_sub_epi8(_mm_max_epi8(ixmm3, ixmm4),
                                         _mm_min_epi8(ixmm3, ixmm4));

            ixmm0 = _mm_add_epi16(_mm_unpacklo_epi8(ixmm2, ixmm_zero),
                                  _mm_unpackhi_epi8(ixmm2, ixmm_zero));
            ixmm1 = _mm_add_epi16(_mm_unpacklo_epi8(ixmm5, ixmm_zero),
                                  _mm_unpackhi_epi8(ixmm5, ixmm_zero));
            ixmm2 = _mm_add_epi16(ixmm0, ixmm1);

            ixmm_sum1 =
                _mm_add_epi32(_mm_unpacklo_epi16(ixmm2, ixmm_zero), ixmm_sum1);
            ixmm_sum2 =
                _mm_add_epi32(_mm_unpackhi_epi16(ixmm2, ixmm_zero), ixmm_sum2);
        }

        if ((last - last_aligned) > 15) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));
            __m128i ixmm3 = _mm_add_epi16(_mm_unpacklo_epi8(ixmm2, ixmm_zero),
                                          _mm_unpackhi_epi8(ixmm2, ixmm_zero));
            ixmm_sum1 =
                _mm_add_epi32(_mm_unpacklo_epi16(ixmm3, ixmm_zero), ixmm_sum1);
            ixmm_sum2 =
                _mm_add_epi32(_mm_unpackhi_epi16(ixmm3, ixmm_zero), ixmm_sum2);
            lhs += 16;
            rhs += 16;
        }
    } else {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));

            __m128i ixmm3 = _mm_lddqu_si128((const __m128i *)(lhs + 16));
            __m128i ixmm4 = _mm_lddqu_si128((const __m128i *)(rhs + 16));
            __m128i ixmm5 = _mm_sub_epi8(_mm_max_epi8(ixmm3, ixmm4),
                                         _mm_min_epi8(ixmm3, ixmm4));

            ixmm0 = _mm_add_epi16(_mm_unpacklo_epi8(ixmm2, ixmm_zero),
                                  _mm_unpackhi_epi8(ixmm2, ixmm_zero));
            ixmm1 = _mm_add_epi16(_mm_unpacklo_epi8(ixmm5, ixmm_zero),
                                  _mm_unpackhi_epi8(ixmm5, ixmm_zero));
            ixmm2 = _mm_add_epi16(ixmm0, ixmm1);

            ixmm_sum1 =
                _mm_add_epi32(_mm_unpacklo_epi16(ixmm2, ixmm_zero), ixmm_sum1);
            ixmm_sum2 =
                _mm_add_epi32(_mm_unpackhi_epi16(ixmm2, ixmm_zero), ixmm_sum2);
        }

        if ((last - last_aligned) > 15) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));
            __m128i ixmm3 = _mm_add_epi16(_mm_unpacklo_epi8(ixmm2, ixmm_zero),
                                          _mm_unpackhi_epi8(ixmm2, ixmm_zero));
            ixmm_sum1 =
                _mm_add_epi32(_mm_unpacklo_epi16(ixmm3, ixmm_zero), ixmm_sum1);
            ixmm_sum2 =
                _mm_add_epi32(_mm_unpackhi_epi16(ixmm3, ixmm_zero), ixmm_sum2);
            lhs += 16;
            rhs += 16;
        }
    }
    result = horizontal_add_v128(_mm_add_epi32(ixmm_sum1, ixmm_sum2));

    switch (last - lhs) {
    case 15:
        result += fast_abs(lhs[14] - rhs[14]);
        /* FALLTHRU */
    case 14:
        result += fast_abs(lhs[13] - rhs[13]);
        /* FALLTHRU */
    case 13:
        result += fast_abs(lhs[12] - rhs[12]);
        /* FALLTHRU */
    case 12:
        result += fast_abs(lhs[11] - rhs[11]);
        /* FALLTHRU */
    case 11:
        result += fast_abs(lhs[10] - rhs[10]);
        /* FALLTHRU */
    case 10:
        result += fast_abs(lhs[9] - rhs[9]);
        /* FALLTHRU */
    case 9:
        result += fast_abs(lhs[8] - rhs[8]);
        /* FALLTHRU */
    case 8:
        result += fast_abs(lhs[7] - rhs[7]);
        /* FALLTHRU */
    case 7:
        result += fast_abs(lhs[6] - rhs[6]);
        /* FALLTHRU */
    case 6:
        result += fast_abs(lhs[5] - rhs[5]);
        /* FALLTHRU */
    case 5:
        result += fast_abs(lhs[4] - rhs[4]);
        /* FALLTHRU */
    case 4:
        result += fast_abs(lhs[3] - rhs[3]);
        /* FALLTHRU */
    case 3:
        result += fast_abs(lhs[2] - rhs[2]);
        /* FALLTHRU */
    case 2:
        result += fast_abs(lhs[1] - rhs[1]);
        /* FALLTHRU */
    case 1:
        result += fast_abs(lhs[0] - rhs[0]);
    }
    return result;
}

static inline float chebyshev_distance_v128(const int8_t *lhs,
                                            const int8_t *rhs, size_t size)
{
    const int8_t *last = lhs + size;
    const int8_t *last_aligned = lhs + ((size >> 5) << 5);
    int32_t result = 0;

    static const __m128i ixmm_zero = _mm_setzero_si128();
    __m128i ixmm_max = _mm_setzero_si128();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));

            __m128i ixmm3 = _mm_load_si128((const __m128i *)(lhs + 16));
            __m128i ixmm4 = _mm_load_si128((const __m128i *)(rhs + 16));
            __m128i ixmm5 = _mm_sub_epi8(_mm_max_epi8(ixmm3, ixmm4),
                                         _mm_min_epi8(ixmm3, ixmm4));

            ixmm_max = _mm_max_epu8(_mm_max_epu8(ixmm2, ixmm5), ixmm_max);
        }

        if ((last - last_aligned) > 15) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));
            ixmm_max = _mm_max_epu8(ixmm2, ixmm_max);
            lhs += 16;
            rhs += 16;
        }
    } else {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));

            __m128i ixmm3 = _mm_lddqu_si128((const __m128i *)(lhs + 16));
            __m128i ixmm4 = _mm_lddqu_si128((const __m128i *)(rhs + 16));
            __m128i ixmm5 = _mm_sub_epi8(_mm_max_epi8(ixmm3, ixmm4),
                                         _mm_min_epi8(ixmm3, ixmm4));

            ixmm_max = _mm_max_epu8(_mm_max_epu8(ixmm2, ixmm5), ixmm_max);
        }

        if ((last - last_aligned) > 15) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_sub_epi8(_mm_max_epi8(ixmm0, ixmm1),
                                         _mm_min_epi8(ixmm0, ixmm1));
            ixmm_max = _mm_max_epu8(ixmm2, ixmm_max);
            lhs += 16;
            rhs += 16;
        }
    }
    ixmm_max = _mm_max_epi16(_mm_unpacklo_epi8(ixmm_max, ixmm_zero),
                             _mm_unpackhi_epi8(ixmm_max, ixmm_zero));
    result = horizontal_max_v128(
        _mm_max_epi32(_mm_unpacklo_epi16(ixmm_max, ixmm_zero),
                      _mm_unpackhi_epi16(ixmm_max, ixmm_zero)));

    int32_t x;
    switch (last - lhs) {
    case 15:
        x = fast_abs(lhs[14] - rhs[14]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 14:
        x = fast_abs(lhs[13] - rhs[13]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 13:
        x = fast_abs(lhs[12] - rhs[12]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 12:
        x = fast_abs(lhs[11] - rhs[11]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 11:
        x = fast_abs(lhs[10] - rhs[10]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 10:
        x = fast_abs(lhs[9] - rhs[9]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 9:
        x = fast_abs(lhs[8] - rhs[8]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 8:
        x = fast_abs(lhs[7] - rhs[7]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 7:
        x = fast_abs(lhs[6] - rhs[6]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 6:
        x = fast_abs(lhs[5] - rhs[5]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 5:
        x = fast_abs(lhs[4] - rhs[4]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 4:
        x = fast_abs(lhs[3] - rhs[3]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 3:
        x = fast_abs(lhs[2] - rhs[2]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 2:
        x = fast_abs(lhs[1] - rhs[1]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 1:
        x = fast_abs(lhs[0] - rhs[0]);
        if (result < x) {
            result = x;
        }
    }
    return result;
}

static inline float cosine_distance_v128(const int8_t *lhs, const int8_t *rhs,
                                         size_t size)
{
    const int8_t *last = lhs + size;
    const int8_t *last_aligned = lhs + ((size >> 4) << 4);
    int32_t sum1 = 0;
    int32_t sum2 = 0;
    int32_t sum3 = 0;

    __m128i ixmm_sum1 = _mm_setzero_si128();
    __m128i ixmm_sum2 = _mm_setzero_si128();
    __m128i ixmm_sum3 = _mm_setzero_si128();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_cvtepi8_epi16(ixmm0);
            __m128i ixmm3 = _mm_cvtepi8_epi16(ixmm1);
            __m128i ixmm4 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm0, ixmm0));
            __m128i ixmm5 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm1, ixmm1));

            ixmm_sum2 =
                _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(ixmm2, ixmm2),
                                            _mm_madd_epi16(ixmm4, ixmm4)),
                              ixmm_sum2);
            ixmm_sum3 =
                _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(ixmm3, ixmm3),
                                            _mm_madd_epi16(ixmm5, ixmm5)),
                              ixmm_sum3);
            ixmm_sum1 =
                _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(ixmm2, ixmm3),
                                            _mm_madd_epi16(ixmm4, ixmm5)),
                              ixmm_sum1);
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_cvtepi8_epi16(ixmm0);
            __m128i ixmm3 = _mm_cvtepi8_epi16(ixmm1);
            __m128i ixmm4 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm0, ixmm0));
            __m128i ixmm5 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm1, ixmm1));

            ixmm_sum2 =
                _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(ixmm2, ixmm2),
                                            _mm_madd_epi16(ixmm4, ixmm4)),
                              ixmm_sum2);
            ixmm_sum3 =
                _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(ixmm3, ixmm3),
                                            _mm_madd_epi16(ixmm5, ixmm5)),
                              ixmm_sum3);
            ixmm_sum1 =
                _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(ixmm2, ixmm3),
                                            _mm_madd_epi16(ixmm4, ixmm5)),
                              ixmm_sum1);
        }
    }
    sum1 = horizontal_add_v128(ixmm_sum1);
    sum2 = horizontal_add_v128(ixmm_sum2);
    sum3 = horizontal_add_v128(ixmm_sum3);

    int32_t x1, x2;
    switch (last - last_aligned) {
    case 15:
        x1 = lhs[14];
        x2 = rhs[14];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 14:
        x1 = lhs[13];
        x2 = rhs[13];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 13:
        x1 = lhs[12];
        x2 = rhs[12];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 12:
        x1 = lhs[11];
        x2 = rhs[11];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 11:
        x1 = lhs[10];
        x2 = rhs[10];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 10:
        x1 = lhs[9];
        x2 = rhs[9];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 9:
        x1 = lhs[8];
        x2 = rhs[8];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 8:
        x1 = lhs[7];
        x2 = rhs[7];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 7:
        x1 = lhs[6];
        x2 = rhs[6];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 6:
        x1 = lhs[5];
        x2 = rhs[5];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 5:
        x1 = lhs[4];
        x2 = rhs[4];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 4:
        x1 = lhs[3];
        x2 = rhs[3];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 3:
        x1 = lhs[2];
        x2 = rhs[2];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1];
        x2 = rhs[1];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0];
        x2 = rhs[0];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
    }
    return (1.0f - (float)sum1 / (fast_sqrt(sum2) * fast_sqrt(sum3)));
}

static inline float inner_product_v128(const int8_t *lhs, const int8_t *rhs,
                                       size_t size)
{
    const int8_t *last = lhs + size;
    const int8_t *last_aligned = lhs + ((size >> 5) << 5);
    int32_t sum = 0;

    __m128i ixmm_sum1 = _mm_setzero_si128();
    __m128i ixmm_sum2 = _mm_setzero_si128();

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_madd_epi16(_mm_cvtepi8_epi16(ixmm0),
                                           _mm_cvtepi8_epi16(ixmm1));
            __m128i ixmm3 = _mm_madd_epi16(
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm0, ixmm0)),
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm1, ixmm1)));

            __m128i ixmm4 = _mm_load_si128((const __m128i *)(lhs + 16));
            __m128i ixmm5 = _mm_load_si128((const __m128i *)(rhs + 16));
            ixmm0 = _mm_madd_epi16(_mm_cvtepi8_epi16(ixmm4),
                                   _mm_cvtepi8_epi16(ixmm5));
            ixmm1 = _mm_madd_epi16(
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm4, ixmm4)),
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm5, ixmm5)));

            ixmm_sum1 = _mm_add_epi32(_mm_add_epi32(ixmm0, ixmm1), ixmm_sum1);
            ixmm_sum2 = _mm_add_epi32(_mm_add_epi32(ixmm2, ixmm3), ixmm_sum2);
        }

        if ((last - last_aligned) > 15) {
            __m128i ixmm0 = _mm_load_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_load_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_madd_epi16(_mm_cvtepi8_epi16(ixmm0),
                                           _mm_cvtepi8_epi16(ixmm1));
            __m128i ixmm3 = _mm_madd_epi16(
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm0, ixmm0)),
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm1, ixmm1)));

            ixmm_sum2 = _mm_add_epi32(_mm_add_epi32(ixmm2, ixmm3), ixmm_sum2);
            lhs += 16;
            rhs += 16;
        }
    } else {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_madd_epi16(_mm_cvtepi8_epi16(ixmm0),
                                           _mm_cvtepi8_epi16(ixmm1));
            __m128i ixmm3 = _mm_madd_epi16(
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm0, ixmm0)),
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm1, ixmm1)));

            __m128i ixmm4 = _mm_lddqu_si128((const __m128i *)(lhs + 16));
            __m128i ixmm5 = _mm_lddqu_si128((const __m128i *)(rhs + 16));
            ixmm0 = _mm_madd_epi16(_mm_cvtepi8_epi16(ixmm4),
                                   _mm_cvtepi8_epi16(ixmm5));
            ixmm1 = _mm_madd_epi16(
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm4, ixmm4)),
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm5, ixmm5)));

            ixmm_sum1 = _mm_add_epi32(_mm_add_epi32(ixmm0, ixmm1), ixmm_sum1);
            ixmm_sum2 = _mm_add_epi32(_mm_add_epi32(ixmm2, ixmm3), ixmm_sum2);
        }

        if ((last - last_aligned) > 15) {
            __m128i ixmm0 = _mm_lddqu_si128((const __m128i *)lhs);
            __m128i ixmm1 = _mm_lddqu_si128((const __m128i *)rhs);
            __m128i ixmm2 = _mm_madd_epi16(_mm_cvtepi8_epi16(ixmm0),
                                           _mm_cvtepi8_epi16(ixmm1));
            __m128i ixmm3 = _mm_madd_epi16(
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm0, ixmm0)),
                _mm_cvtepi8_epi16(_mm_unpackhi_epi64(ixmm1, ixmm1)));

            ixmm_sum2 = _mm_add_epi32(_mm_add_epi32(ixmm2, ixmm3), ixmm_sum2);
            lhs += 16;
            rhs += 16;
        }
    }
    sum = horizontal_add_v128(_mm_add_epi32(ixmm_sum1, ixmm_sum2));

    switch (last - lhs) {
    case 15:
        sum += (lhs[14] * rhs[14]);
        /* FALLTHRU */
    case 14:
        sum += (lhs[13] * rhs[13]);
        /* FALLTHRU */
    case 13:
        sum += (lhs[12] * rhs[12]);
        /* FALLTHRU */
    case 12:
        sum += (lhs[11] * rhs[11]);
        /* FALLTHRU */
    case 11:
        sum += (lhs[10] * rhs[10]);
        /* FALLTHRU */
    case 10:
        sum += (lhs[9] * rhs[9]);
        /* FALLTHRU */
    case 9:
        sum += (lhs[8] * rhs[8]);
        /* FALLTHRU */
    case 8:
        sum += (lhs[7] * rhs[7]);
        /* FALLTHRU */
    case 7:
        sum += (lhs[6] * rhs[6]);
        /* FALLTHRU */
    case 6:
        sum += (lhs[5] * rhs[5]);
        /* FALLTHRU */
    case 5:
        sum += (lhs[4] * rhs[4]);
        /* FALLTHRU */
    case 4:
        sum += (lhs[3] * rhs[3]);
        /* FALLTHRU */
    case 3:
        sum += (lhs[2] * rhs[2]);
        /* FALLTHRU */
    case 2:
        sum += (lhs[1] * rhs[1]);
        /* FALLTHRU */
    case 1:
        sum += (lhs[0] * rhs[0]);
    }
    return sum;
}
#endif // __SSE4_1__

#if defined(__AVX__)
#ifndef __FMA__
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
#endif

static inline float horizontal_add_v256(__m256 v)
{
    __m256 x1 = _mm256_hadd_ps(v, v);
    __m256 x2 = _mm256_hadd_ps(x1, x1);
    __m128 x3 = _mm256_extractf128_ps(x2, 1);
    __m128 x4 = _mm_add_ss(_mm256_castps256_ps128(x2), x3);
    return _mm_cvtss_f32(x4);
}

static inline float horizontal_max_v256(__m256 v)
{
    __m256 x1 = _mm256_permute_ps(v, _MM_SHUFFLE(0, 0, 3, 2));
    __m256 x2 = _mm256_max_ps(v, x1);
    __m256 x3 = _mm256_permute_ps(x2, _MM_SHUFFLE(0, 0, 0, 1));
    __m256 x4 = _mm256_max_ps(x2, x3);
    __m128 x5 = _mm256_extractf128_ps(x4, 1);
    __m128 x6 = _mm_max_ss(_mm256_castps256_ps128(x4), x5);
    return _mm_cvtss_f32(x6);
}

static inline float horizontal_mean_v256(const float *lhs, size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);
    float sum = 0.0f;

    __m256 ymm_sum1 = _mm256_setzero_ps();
    __m256 ymm_sum2 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16) {
            ymm_sum1 = _mm256_add_ps(_mm256_load_ps(lhs), ymm_sum1);
            ymm_sum2 = _mm256_add_ps(_mm256_load_ps(lhs + 8), ymm_sum2);
        }

        if ((last - last_aligned) > 7) {
            ymm_sum1 = _mm256_add_ps(_mm256_load_ps(lhs), ymm_sum1);
            lhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16) {
            ymm_sum1 = _mm256_add_ps(_mm256_loadu_ps(lhs), ymm_sum1);
            ymm_sum2 = _mm256_add_ps(_mm256_loadu_ps(lhs + 8), ymm_sum2);
        }

        if ((last - last_aligned) > 7) {
            ymm_sum1 = _mm256_add_ps(_mm256_loadu_ps(lhs), ymm_sum1);
            lhs += 8;
        }
    }
    sum = horizontal_add_v256(_mm256_add_ps(ymm_sum1, ymm_sum2));

    switch (last - lhs) {
    case 7:
        sum += lhs[6];
        /* FALLTHRU */
    case 6:
        sum += lhs[5];
        /* FALLTHRU */
    case 5:
        sum += lhs[4];
        /* FALLTHRU */
    case 4:
        sum += lhs[3];
        /* FALLTHRU */
    case 3:
        sum += lhs[2];
        /* FALLTHRU */
    case 2:
        sum += lhs[1];
        /* FALLTHRU */
    case 1:
        sum += lhs[0];
    }
    return (sum / (float)size);
}

static inline float
squared_euclidean_distance_v256(const float *lhs, const float *rhs, size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);
    float result = 0.0f;

    __m256 ymm_sum1 = _mm256_setzero_ps();
    __m256 ymm_sum2 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
            __m256 ymm1 =
                _mm256_sub_ps(_mm256_load_ps(lhs + 8), _mm256_load_ps(rhs + 8));
            ymm_sum1 = _mm256_fmadd_ps(ymm0, ymm0, ymm_sum1);
            ymm_sum2 = _mm256_fmadd_ps(ymm1, ymm1, ymm_sum2);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
            ymm_sum1 = _mm256_fmadd_ps(ymm0, ymm0, ymm_sum1);
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
            __m256 ymm1 = _mm256_sub_ps(_mm256_loadu_ps(lhs + 8),
                                        _mm256_loadu_ps(rhs + 8));
            ymm_sum1 = _mm256_fmadd_ps(ymm0, ymm0, ymm_sum1);
            ymm_sum2 = _mm256_fmadd_ps(ymm1, ymm1, ymm_sum2);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
            ymm_sum1 = _mm256_fmadd_ps(ymm0, ymm0, ymm_sum1);
            lhs += 8;
            rhs += 8;
        }
    }
    result = horizontal_add_v256(_mm256_add_ps(ymm_sum1, ymm_sum2));

    float x;
    switch (last - lhs) {
    case 7:
        x = lhs[6] - rhs[6];
        result += (x * x);
        /* FALLTHRU */
    case 6:
        x = lhs[5] - rhs[5];
        result += (x * x);
        /* FALLTHRU */
    case 5:
        x = lhs[4] - rhs[4];
        result += (x * x);
        /* FALLTHRU */
    case 4:
        x = lhs[3] - rhs[3];
        result += (x * x);
        /* FALLTHRU */
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x);
    }
    return result;
}

static inline float normalized_squared_euclidean_distance_v256(const float *lhs,
                                                               const float *rhs,
                                                               size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float avg1 = horizontal_mean_v256(lhs, size);
    float avg2 = horizontal_mean_v256(rhs, size);

    __m256 ymm_sum11 = _mm256_setzero_ps();
    __m256 ymm_sum12 = _mm256_setzero_ps();
    __m256 ymm_sum13 = _mm256_setzero_ps();
    __m256 ymm_sum21 = _mm256_setzero_ps();
    __m256 ymm_sum22 = _mm256_setzero_ps();
    __m256 ymm_sum23 = _mm256_setzero_ps();
    __m256 ymm_avg1 = _mm256_set1_ps(avg1);
    __m256 ymm_avg2 = _mm256_set1_ps(avg2);

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm11 = _mm256_sub_ps(_mm256_load_ps(lhs), ymm_avg1);
            __m256 ymm12 = _mm256_sub_ps(_mm256_load_ps(rhs), ymm_avg2);
            __m256 ymm21 = _mm256_sub_ps(_mm256_load_ps(lhs + 8), ymm_avg1);
            __m256 ymm22 = _mm256_sub_ps(_mm256_load_ps(rhs + 8), ymm_avg2);
            __m256 ymm13 = _mm256_sub_ps(ymm11, ymm12);
            __m256 ymm23 = _mm256_sub_ps(ymm21, ymm22);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm13, ymm13, ymm_sum13);
            ymm_sum21 = _mm256_fmadd_ps(ymm21, ymm21, ymm_sum21);
            ymm_sum22 = _mm256_fmadd_ps(ymm22, ymm22, ymm_sum22);
            ymm_sum23 = _mm256_fmadd_ps(ymm23, ymm23, ymm_sum23);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm11 = _mm256_sub_ps(_mm256_load_ps(lhs), ymm_avg1);
            __m256 ymm12 = _mm256_sub_ps(_mm256_load_ps(rhs), ymm_avg2);
            __m256 ymm13 = _mm256_sub_ps(ymm11, ymm12);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm13, ymm13, ymm_sum13);
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm11 = _mm256_sub_ps(_mm256_loadu_ps(lhs), ymm_avg1);
            __m256 ymm12 = _mm256_sub_ps(_mm256_loadu_ps(rhs), ymm_avg2);
            __m256 ymm21 = _mm256_sub_ps(_mm256_loadu_ps(lhs + 8), ymm_avg1);
            __m256 ymm22 = _mm256_sub_ps(_mm256_loadu_ps(rhs + 8), ymm_avg2);
            __m256 ymm13 = _mm256_sub_ps(ymm11, ymm12);
            __m256 ymm23 = _mm256_sub_ps(ymm21, ymm22);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm13, ymm13, ymm_sum13);
            ymm_sum21 = _mm256_fmadd_ps(ymm21, ymm21, ymm_sum21);
            ymm_sum22 = _mm256_fmadd_ps(ymm22, ymm22, ymm_sum22);
            ymm_sum23 = _mm256_fmadd_ps(ymm23, ymm23, ymm_sum23);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm11 = _mm256_sub_ps(_mm256_loadu_ps(lhs), ymm_avg1);
            __m256 ymm12 = _mm256_sub_ps(_mm256_loadu_ps(rhs), ymm_avg2);
            __m256 ymm13 = _mm256_sub_ps(ymm11, ymm12);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm13, ymm13, ymm_sum13);
            lhs += 8;
            rhs += 8;
        }
    }
    sum1 = horizontal_add_v256(_mm256_add_ps(ymm_sum13, ymm_sum23));
    sum2 =
        horizontal_add_v256(_mm256_add_ps(_mm256_add_ps(ymm_sum11, ymm_sum21),
                                          _mm256_add_ps(ymm_sum12, ymm_sum22)));

    float x1, x2, x3;
    switch (last - lhs) {
    case 7:
        x1 = lhs[6] - avg1;
        x2 = rhs[6] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
        /* FALLTHRU */
    case 6:
        x1 = lhs[5] - avg1;
        x2 = rhs[5] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
        /* FALLTHRU */
    case 5:
        x1 = lhs[4] - avg1;
        x2 = rhs[4] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
        /* FALLTHRU */
    case 4:
        x1 = lhs[3] - avg1;
        x2 = rhs[3] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
        /* FALLTHRU */
    case 3:
        x1 = lhs[2] - avg1;
        x2 = rhs[2] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1] - avg1;
        x2 = rhs[1] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0] - avg1;
        x2 = rhs[0] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
    }
    return (sum1 / sum2 / 2.0f);
}

static inline float weighted_squared_euclidean_distance_v256(const float *lhs,
                                                             const float *rhs,
                                                             const float *wgt,
                                                             size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);
    float result = 0.0f;

    __m256 ymm_sum1 = _mm256_setzero_ps();
    __m256 ymm_sum2 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0 &&
        ((uintptr_t)wgt & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16, wgt += 16) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
            __m256 ymm1 =
                _mm256_sub_ps(_mm256_load_ps(lhs + 8), _mm256_load_ps(rhs + 8));
            ymm_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(ymm0, ymm0),
                                       _mm256_load_ps(wgt), ymm_sum1);
            ymm_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(ymm1, ymm1),
                                       _mm256_load_ps(wgt + 8), ymm_sum2);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
            ymm_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(ymm0, ymm0),
                                       _mm256_load_ps(wgt), ymm_sum1);
            lhs += 8;
            rhs += 8;
            wgt += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16, wgt += 16) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
            __m256 ymm1 = _mm256_sub_ps(_mm256_loadu_ps(lhs + 8),
                                        _mm256_loadu_ps(rhs + 8));
            ymm_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(ymm0, ymm0),
                                       _mm256_loadu_ps(wgt), ymm_sum1);
            ymm_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(ymm1, ymm1),
                                       _mm256_loadu_ps(wgt + 8), ymm_sum2);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
            ymm_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(ymm0, ymm0),
                                       _mm256_loadu_ps(wgt), ymm_sum1);
            lhs += 8;
            rhs += 8;
            wgt += 8;
        }
    }
    result = horizontal_add_v256(_mm256_add_ps(ymm_sum1, ymm_sum2));

    float x;
    switch (last - lhs) {
    case 7:
        x = lhs[6] - rhs[6];
        result += (x * x * wgt[6]);
        /* FALLTHRU */
    case 6:
        x = lhs[5] - rhs[5];
        result += (x * x * wgt[5]);
        /* FALLTHRU */
    case 5:
        x = lhs[4] - rhs[4];
        result += (x * x * wgt[4]);
        /* FALLTHRU */
    case 4:
        x = lhs[3] - rhs[3];
        result += (x * x * wgt[3]);
        /* FALLTHRU */
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x * wgt[2]);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x * wgt[1]);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x * wgt[0]);
    }
    return result;
}

static inline float manhattan_distance_v256(const float *lhs, const float *rhs,
                                            size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);
    float result = 0.0f;

    static const __m256 ymm_mask =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffffu));
    __m256 ymm_sum1 = _mm256_setzero_ps();
    __m256 ymm_sum2 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
            __m256 ymm1 =
                _mm256_sub_ps(_mm256_load_ps(lhs + 8), _mm256_load_ps(rhs + 8));
            ymm_sum1 = _mm256_add_ps(ymm_sum1, _mm256_and_ps(ymm_mask, ymm0));
            ymm_sum2 = _mm256_add_ps(ymm_sum2, _mm256_and_ps(ymm_mask, ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
            ymm_sum1 = _mm256_add_ps(ymm_sum1, _mm256_and_ps(ymm_mask, ymm0));
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
            __m256 ymm1 = _mm256_sub_ps(_mm256_loadu_ps(lhs + 8),
                                        _mm256_loadu_ps(rhs + 8));
            ymm_sum1 = _mm256_add_ps(ymm_sum1, _mm256_and_ps(ymm_mask, ymm0));
            ymm_sum2 = _mm256_add_ps(ymm_sum2, _mm256_and_ps(ymm_mask, ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
            ymm_sum1 = _mm256_add_ps(ymm_sum1, _mm256_and_ps(ymm_mask, ymm0));
            lhs += 8;
            rhs += 8;
        }
    }
    result = horizontal_add_v256(_mm256_add_ps(ymm_sum1, ymm_sum2));

    switch (last - lhs) {
    case 7:
        result += fast_abs(lhs[6] - rhs[6]);
        /* FALLTHRU */
    case 6:
        result += fast_abs(lhs[5] - rhs[5]);
        /* FALLTHRU */
    case 5:
        result += fast_abs(lhs[4] - rhs[4]);
        /* FALLTHRU */
    case 4:
        result += fast_abs(lhs[3] - rhs[3]);
        /* FALLTHRU */
    case 3:
        result += fast_abs(lhs[2] - rhs[2]);
        /* FALLTHRU */
    case 2:
        result += fast_abs(lhs[1] - rhs[1]);
        /* FALLTHRU */
    case 1:
        result += fast_abs(lhs[0] - rhs[0]);
    }
    return result;
}

static inline float chebyshev_distance_v256(const float *lhs, const float *rhs,
                                            size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);
    float result = 0.0f;

    static const __m256 ymm_mask =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffffu));
    __m256 ymm_max0 = _mm256_setzero_ps();
    __m256 ymm_max1 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
            __m256 ymm1 =
                _mm256_sub_ps(_mm256_load_ps(lhs + 8), _mm256_load_ps(rhs + 8));
            ymm_max0 = _mm256_max_ps(ymm_max0, _mm256_and_ps(ymm_mask, ymm0));
            ymm_max1 = _mm256_max_ps(ymm_max1, _mm256_and_ps(ymm_mask, ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
            ymm_max0 = _mm256_max_ps(ymm_max0, _mm256_and_ps(ymm_mask, ymm0));
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
            __m256 ymm1 = _mm256_sub_ps(_mm256_loadu_ps(lhs + 8),
                                        _mm256_loadu_ps(rhs + 8));
            ymm_max0 = _mm256_max_ps(ymm_max0, _mm256_and_ps(ymm_mask, ymm0));
            ymm_max1 = _mm256_max_ps(ymm_max1, _mm256_and_ps(ymm_mask, ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm0 =
                _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
            ymm_max0 = _mm256_max_ps(ymm_max0, _mm256_and_ps(ymm_mask, ymm0));
            lhs += 8;
            rhs += 8;
        }
    }
    result = horizontal_max_v256(_mm256_max_ps(ymm_max0, ymm_max1));

    float x;
    switch (last - lhs) {
    case 7:
        x = fast_abs(lhs[6] - rhs[6]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 6:
        x = fast_abs(lhs[5] - rhs[5]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 5:
        x = fast_abs(lhs[4] - rhs[4]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 4:
        x = fast_abs(lhs[3] - rhs[3]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 3:
        x = fast_abs(lhs[2] - rhs[2]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 2:
        x = fast_abs(lhs[1] - rhs[1]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 1:
        x = fast_abs(lhs[0] - rhs[0]);
        if (result < x) {
            result = x;
        }
    }
    return result;
}

static inline float cosine_distance_v256(const float *lhs, const float *rhs,
                                         size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    __m256 ymm_sum11 = _mm256_setzero_ps();
    __m256 ymm_sum12 = _mm256_setzero_ps();
    __m256 ymm_sum13 = _mm256_setzero_ps();
    __m256 ymm_sum21 = _mm256_setzero_ps();
    __m256 ymm_sum22 = _mm256_setzero_ps();
    __m256 ymm_sum23 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm11 = _mm256_load_ps(lhs);
            __m256 ymm21 = _mm256_load_ps(lhs + 8);
            __m256 ymm12 = _mm256_load_ps(rhs);
            __m256 ymm22 = _mm256_load_ps(rhs + 8);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm12, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum13);
            ymm_sum21 = _mm256_fmadd_ps(ymm21, ymm22, ymm_sum21);
            ymm_sum22 = _mm256_fmadd_ps(ymm21, ymm21, ymm_sum22);
            ymm_sum23 = _mm256_fmadd_ps(ymm22, ymm22, ymm_sum23);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm11 = _mm256_load_ps(lhs);
            __m256 ymm12 = _mm256_load_ps(rhs);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm12, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum13);
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm11 = _mm256_loadu_ps(lhs);
            __m256 ymm21 = _mm256_loadu_ps(lhs + 8);
            __m256 ymm12 = _mm256_loadu_ps(rhs);
            __m256 ymm22 = _mm256_loadu_ps(rhs + 8);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm12, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum13);
            ymm_sum21 = _mm256_fmadd_ps(ymm21, ymm22, ymm_sum21);
            ymm_sum22 = _mm256_fmadd_ps(ymm21, ymm21, ymm_sum22);
            ymm_sum23 = _mm256_fmadd_ps(ymm22, ymm22, ymm_sum23);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm11 = _mm256_loadu_ps(lhs);
            __m256 ymm12 = _mm256_loadu_ps(rhs);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm12, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum13);
            lhs += 8;
            rhs += 8;
        }
    }
    sum1 = horizontal_add_v256(_mm256_add_ps(ymm_sum11, ymm_sum21));
    sum2 = horizontal_add_v256(_mm256_add_ps(ymm_sum12, ymm_sum22));
    sum3 = horizontal_add_v256(_mm256_add_ps(ymm_sum13, ymm_sum23));

    float x1, x2;
    switch (last - lhs) {
    case 7:
        x1 = lhs[6];
        x2 = rhs[6];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 6:
        x1 = lhs[5];
        x2 = rhs[5];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 5:
        x1 = lhs[4];
        x2 = rhs[4];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 4:
        x1 = lhs[3];
        x2 = rhs[3];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 3:
        x1 = lhs[2];
        x2 = rhs[2];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1];
        x2 = rhs[1];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0];
        x2 = rhs[0];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
    }
    return (1.0f - sum1 / (fast_sqrt(sum2) * fast_sqrt(sum3)));
}

static inline float canberra_distance_v256(const float *lhs, const float *rhs,
                                           size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);
    float result = 0.0f;

    static const __m256 ymm_mask =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffffu));
    __m256 ymm_sum1 = _mm256_setzero_ps();
    __m256 ymm_sum2 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm10 = _mm256_load_ps(lhs);
            __m256 ymm20 = _mm256_load_ps(lhs + 8);
            __m256 ymm11 = _mm256_load_ps(rhs);
            __m256 ymm21 = _mm256_load_ps(rhs + 8);
            __m256 ymm13 = _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm10, ymm11));
            __m256 ymm23 = _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm20, ymm21));
            __m256 ymm14 = _mm256_add_ps(_mm256_and_ps(ymm_mask, ymm10),
                                         _mm256_and_ps(ymm_mask, ymm11));
            __m256 ymm24 = _mm256_add_ps(_mm256_and_ps(ymm_mask, ymm20),
                                         _mm256_and_ps(ymm_mask, ymm21));
            ymm_sum1 = _mm256_add_ps(_mm256_div_ps(ymm13, ymm14), ymm_sum1);
            ymm_sum2 = _mm256_add_ps(_mm256_div_ps(ymm23, ymm24), ymm_sum2);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm10 = _mm256_load_ps(lhs);
            __m256 ymm11 = _mm256_load_ps(rhs);
            __m256 ymm13 = _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm10, ymm11));
            __m256 ymm14 = _mm256_add_ps(_mm256_and_ps(ymm_mask, ymm10),
                                         _mm256_and_ps(ymm_mask, ymm11));
            ymm_sum1 = _mm256_add_ps(_mm256_div_ps(ymm13, ymm14), ymm_sum1);
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm10 = _mm256_loadu_ps(lhs);
            __m256 ymm11 = _mm256_loadu_ps(rhs);
            __m256 ymm20 = _mm256_loadu_ps(lhs + 8);
            __m256 ymm21 = _mm256_loadu_ps(rhs + 8);
            __m256 ymm13 = _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm10, ymm11));
            __m256 ymm14 = _mm256_add_ps(_mm256_and_ps(ymm_mask, ymm10),
                                         _mm256_and_ps(ymm_mask, ymm11));
            __m256 ymm23 = _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm20, ymm21));
            __m256 ymm24 = _mm256_add_ps(_mm256_and_ps(ymm_mask, ymm20),
                                         _mm256_and_ps(ymm_mask, ymm21));
            ymm_sum1 = _mm256_add_ps(_mm256_div_ps(ymm13, ymm14), ymm_sum1);
            ymm_sum2 = _mm256_add_ps(_mm256_div_ps(ymm23, ymm24), ymm_sum2);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm10 = _mm256_loadu_ps(lhs);
            __m256 ymm11 = _mm256_loadu_ps(rhs);
            __m256 ymm13 = _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm10, ymm11));
            __m256 ymm14 = _mm256_add_ps(_mm256_and_ps(ymm_mask, ymm10),
                                         _mm256_and_ps(ymm_mask, ymm11));
            ymm_sum1 = _mm256_add_ps(_mm256_div_ps(ymm13, ymm14), ymm_sum1);
            lhs += 8;
            rhs += 8;
        }
    }
    result = horizontal_add_v256(_mm256_add_ps(ymm_sum1, ymm_sum2));

    float x1, x2;
    switch (last - lhs) {
    case 7:
        x1 = lhs[6];
        x2 = rhs[6];
        result += (fast_abs(x1 - x2) / (fast_abs(x1) + fast_abs(x2)));
        /* FALLTHRU */
    case 6:
        x1 = lhs[5];
        x2 = rhs[5];
        result += (fast_abs(x1 - x2) / (fast_abs(x1) + fast_abs(x2)));
        /* FALLTHRU */
    case 5:
        x1 = lhs[4];
        x2 = rhs[4];
        result += (fast_abs(x1 - x2) / (fast_abs(x1) + fast_abs(x2)));
        /* FALLTHRU */
    case 4:
        x1 = lhs[3];
        x2 = rhs[3];
        result += (fast_abs(x1 - x2) / (fast_abs(x1) + fast_abs(x2)));
        /* FALLTHRU */
    case 3:
        x1 = lhs[2];
        x2 = rhs[2];
        result += (fast_abs(x1 - x2) / (fast_abs(x1) + fast_abs(x2)));
        /* FALLTHRU */
    case 2:
        x1 = lhs[1];
        x2 = rhs[1];
        result += (fast_abs(x1 - x2) / (fast_abs(x1) + fast_abs(x2)));
        /* FALLTHRU */
    case 1:
        x1 = lhs[0];
        x2 = rhs[0];
        result += (fast_abs(x1 - x2) / (fast_abs(x1) + fast_abs(x2)));
    }
    return result;
}

static inline float bray_curtis_distance_v256(const float *lhs,
                                              const float *rhs, size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);
    float sum1 = 0.0f;
    float sum2 = 0.0f;

    static const __m256 ymm_mask =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffffu));
    __m256 ymm_sum11 = _mm256_setzero_ps();
    __m256 ymm_sum12 = _mm256_setzero_ps();
    __m256 ymm_sum21 = _mm256_setzero_ps();
    __m256 ymm_sum22 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm10 = _mm256_load_ps(lhs);
            __m256 ymm20 = _mm256_load_ps(lhs + 8);
            __m256 ymm11 = _mm256_load_ps(rhs);
            __m256 ymm21 = _mm256_load_ps(rhs + 8);
            ymm_sum11 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm10, ymm11)),
                ymm_sum11);
            ymm_sum12 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_add_ps(ymm10, ymm11)),
                ymm_sum12);
            ymm_sum21 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm20, ymm21)),
                ymm_sum21);
            ymm_sum22 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_add_ps(ymm20, ymm21)),
                ymm_sum22);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm10 = _mm256_load_ps(lhs);
            __m256 ymm11 = _mm256_load_ps(rhs);
            ymm_sum11 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm10, ymm11)),
                ymm_sum11);
            ymm_sum12 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_add_ps(ymm10, ymm11)),
                ymm_sum12);
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm10 = _mm256_loadu_ps(lhs);
            __m256 ymm20 = _mm256_loadu_ps(lhs + 8);
            __m256 ymm11 = _mm256_loadu_ps(rhs);
            __m256 ymm21 = _mm256_loadu_ps(rhs + 8);
            ymm_sum11 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm10, ymm11)),
                ymm_sum11);
            ymm_sum12 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_add_ps(ymm10, ymm11)),
                ymm_sum12);
            ymm_sum21 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm20, ymm21)),
                ymm_sum21);
            ymm_sum22 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_add_ps(ymm20, ymm21)),
                ymm_sum22);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm10 = _mm256_loadu_ps(lhs);
            __m256 ymm11 = _mm256_loadu_ps(rhs);
            ymm_sum11 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_sub_ps(ymm10, ymm11)),
                ymm_sum11);
            ymm_sum12 = _mm256_add_ps(
                _mm256_and_ps(ymm_mask, _mm256_add_ps(ymm10, ymm11)),
                ymm_sum12);
            lhs += 8;
            rhs += 8;
        }
    }
    sum1 = horizontal_add_v256(_mm256_add_ps(ymm_sum11, ymm_sum21));
    sum2 = horizontal_add_v256(_mm256_add_ps(ymm_sum12, ymm_sum22));

    float x1, x2;
    switch (last - lhs) {
    case 7:
        x1 = lhs[6];
        x2 = rhs[6];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
        /* FALLTHRU */
    case 6:
        x1 = lhs[5];
        x2 = rhs[5];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
        /* FALLTHRU */
    case 5:
        x1 = lhs[4];
        x2 = rhs[4];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
        /* FALLTHRU */
    case 4:
        x1 = lhs[3];
        x2 = rhs[3];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
        /* FALLTHRU */
    case 3:
        x1 = lhs[2];
        x2 = rhs[2];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1];
        x2 = rhs[1];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0];
        x2 = rhs[0];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
    }
    return (sum1 / sum2);
}

static inline float correlation_distance_v256(const float *lhs,
                                              const float *rhs, size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float avg1 = horizontal_mean_v256(lhs, size);
    float avg2 = horizontal_mean_v256(rhs, size);

    __m256 ymm_sum11 = _mm256_setzero_ps();
    __m256 ymm_sum12 = _mm256_setzero_ps();
    __m256 ymm_sum13 = _mm256_setzero_ps();
    __m256 ymm_sum21 = _mm256_setzero_ps();
    __m256 ymm_sum22 = _mm256_setzero_ps();
    __m256 ymm_sum23 = _mm256_setzero_ps();
    __m256 ymm_avg1 = _mm256_set1_ps(avg1);
    __m256 ymm_avg2 = _mm256_set1_ps(avg2);

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm11 = _mm256_sub_ps(_mm256_load_ps(lhs), ymm_avg1);
            __m256 ymm12 = _mm256_sub_ps(_mm256_load_ps(rhs), ymm_avg2);
            __m256 ymm21 = _mm256_sub_ps(_mm256_load_ps(lhs + 8), ymm_avg1);
            __m256 ymm22 = _mm256_sub_ps(_mm256_load_ps(rhs + 8), ymm_avg2);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm12, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum13);
            ymm_sum21 = _mm256_fmadd_ps(ymm21, ymm22, ymm_sum21);
            ymm_sum22 = _mm256_fmadd_ps(ymm21, ymm21, ymm_sum22);
            ymm_sum23 = _mm256_fmadd_ps(ymm22, ymm22, ymm_sum23);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm11 = _mm256_sub_ps(_mm256_load_ps(lhs), ymm_avg1);
            __m256 ymm12 = _mm256_sub_ps(_mm256_load_ps(rhs), ymm_avg2);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm12, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum13);
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm11 = _mm256_sub_ps(_mm256_loadu_ps(lhs), ymm_avg1);
            __m256 ymm12 = _mm256_sub_ps(_mm256_loadu_ps(rhs), ymm_avg2);
            __m256 ymm21 = _mm256_sub_ps(_mm256_loadu_ps(lhs + 8), ymm_avg1);
            __m256 ymm22 = _mm256_sub_ps(_mm256_loadu_ps(rhs + 8), ymm_avg2);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm12, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum13);
            ymm_sum21 = _mm256_fmadd_ps(ymm21, ymm22, ymm_sum21);
            ymm_sum22 = _mm256_fmadd_ps(ymm21, ymm21, ymm_sum22);
            ymm_sum23 = _mm256_fmadd_ps(ymm22, ymm22, ymm_sum23);
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm11 = _mm256_sub_ps(_mm256_loadu_ps(lhs), ymm_avg1);
            __m256 ymm12 = _mm256_sub_ps(_mm256_loadu_ps(rhs), ymm_avg2);
            ymm_sum11 = _mm256_fmadd_ps(ymm11, ymm12, ymm_sum11);
            ymm_sum12 = _mm256_fmadd_ps(ymm11, ymm11, ymm_sum12);
            ymm_sum13 = _mm256_fmadd_ps(ymm12, ymm12, ymm_sum13);
            lhs += 8;
            rhs += 8;
        }
    }
    sum1 = horizontal_add_v256(_mm256_add_ps(ymm_sum11, ymm_sum21));
    sum2 = horizontal_add_v256(_mm256_add_ps(ymm_sum12, ymm_sum22));
    sum3 = horizontal_add_v256(_mm256_add_ps(ymm_sum13, ymm_sum23));

    float x1, x2;
    switch (last - lhs) {
    case 7:
        x1 = lhs[6] - avg1;
        x2 = rhs[6] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 6:
        x1 = lhs[5] - avg1;
        x2 = rhs[5] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 5:
        x1 = lhs[4] - avg1;
        x2 = rhs[4] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 4:
        x1 = lhs[3] - avg1;
        x2 = rhs[3] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 3:
        x1 = lhs[2] - avg1;
        x2 = rhs[2] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1] - avg1;
        x2 = rhs[1] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0] - avg1;
        x2 = rhs[0] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
    }
    return (1.0f - sum1 / (fast_sqrt(sum2) * fast_sqrt(sum3)));
}

static inline float binary_distance_v256(const float *lhs, const float *rhs,
                                         size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm0 =
                _mm256_cmp_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs), 0);
            __m256 ymm1 = _mm256_cmp_ps(_mm256_load_ps(lhs + 8),
                                        _mm256_load_ps(rhs + 8), 0);

            if (_mm256_movemask_ps(ymm0) != 0xff ||
                _mm256_movemask_ps(ymm1) != 0xff) {
                return 1.0f;
            }
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm0 =
                _mm256_cmp_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs), 0);
            if (_mm256_movemask_ps(ymm0) != 0xff) {
                return 1.0f;
            }
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256 ymm0 =
                _mm256_cmp_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs), 0);
            __m256 ymm1 = _mm256_cmp_ps(_mm256_loadu_ps(lhs + 8),
                                        _mm256_loadu_ps(rhs + 8), 0);

            if (_mm256_movemask_ps(ymm0) != 0xff ||
                _mm256_movemask_ps(ymm1) != 0xff) {
                return 1.0f;
            }
        }

        if ((last - last_aligned) > 7) {
            __m256 ymm0 =
                _mm256_cmp_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs), 0);
            if (_mm256_movemask_ps(ymm0) != 0xff) {
                return 1.0f;
            }
            lhs += 8;
            rhs += 8;
        }
    }

    switch (last - lhs) {
    case 7:
        if (fast_abs(lhs[6] - rhs[6]) > FLT_EPSILON) {
            return 1.0f;
        }
        /* FALLTHRU */
    case 6:
        if (fast_abs(lhs[5] - rhs[5]) > FLT_EPSILON) {
            return 1.0f;
        }
        /* FALLTHRU */
    case 5:
        if (fast_abs(lhs[4] - rhs[4]) > FLT_EPSILON) {
            return 1.0f;
        }
        /* FALLTHRU */
    case 4:
        if (fast_abs(lhs[3] - rhs[3]) > FLT_EPSILON) {
            return 1.0f;
        }
        /* FALLTHRU */
    case 3:
        if (fast_abs(lhs[2] - rhs[2]) > FLT_EPSILON) {
            return 1.0f;
        }
        /* FALLTHRU */
    case 2:
        if (fast_abs(lhs[1] - rhs[1]) > FLT_EPSILON) {
            return 1.0f;
        }
        /* FALLTHRU */
    case 1:
        if (fast_abs(lhs[0] - rhs[0]) > FLT_EPSILON) {
            return 1.0f;
        }
    }
    return 0.0f;
}

static inline float inner_product_v256(const float *lhs, const float *rhs,
                                       size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 4) << 4);
    float sum = 0.0f;

    __m256 ymm_sum1 = _mm256_setzero_ps();
    __m256 ymm_sum2 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            ymm_sum1 = _mm256_fmadd_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs),
                                       ymm_sum1);
            ymm_sum2 = _mm256_fmadd_ps(_mm256_load_ps(lhs + 8),
                                       _mm256_load_ps(rhs + 8), ymm_sum2);
        }

        if ((last - last_aligned) > 7) {
            ymm_sum1 = _mm256_fmadd_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs),
                                       ymm_sum1);
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            ymm_sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs),
                                       _mm256_loadu_ps(rhs), ymm_sum1);
            ymm_sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + 8),
                                       _mm256_loadu_ps(rhs + 8), ymm_sum2);
        }

        if ((last - last_aligned) > 7) {
            ymm_sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs),
                                       _mm256_loadu_ps(rhs), ymm_sum1);
            lhs += 8;
            rhs += 8;
        }
    }
    sum = horizontal_add_v256(_mm256_add_ps(ymm_sum1, ymm_sum2));

    switch (last - lhs) {
    case 7:
        sum += (lhs[6] * rhs[6]);
        /* FALLTHRU */
    case 6:
        sum += (lhs[5] * rhs[5]);
        /* FALLTHRU */
    case 5:
        sum += (lhs[4] * rhs[4]);
        /* FALLTHRU */
    case 4:
        sum += (lhs[3] * rhs[3]);
        /* FALLTHRU */
    case 3:
        sum += (lhs[2] * rhs[2]);
        /* FALLTHRU */
    case 2:
        sum += (lhs[1] * rhs[1]);
        /* FALLTHRU */
    case 1:
        sum += (lhs[0] * rhs[0]);
    }
    return sum;
}
#endif // __AVX__

#if defined(__AVX2__)
static inline int32_t horizontal_add_v256(__m256i v)
{
    __m256i x1 = _mm256_hadd_epi32(v, v);
    __m256i x2 = _mm256_hadd_epi32(x1, x1);
#if defined(_MSC_VER) && _MSC_VER <= 1700 && !defined(__INTEL_COMPILER)
    __m128i x3 = _mm256_extractf128_si256(x2, 1);
#else
    __m128i x3 = _mm256_extracti128_si256(x2, 1);
#endif
    __m128i x4 = _mm_add_epi32(_mm256_castsi256_si128(x2), x3);
    return _mm_cvtsi128_si32(x4);
}

static inline int32_t horizontal_max_v256(__m256i v)
{
    __m256i x1 = _mm256_shuffle_epi32(v, _MM_SHUFFLE(0, 0, 3, 2));
    __m256i x2 = _mm256_max_epi32(v, x1);
    __m256i x3 = _mm256_shuffle_epi32(x2, _MM_SHUFFLE(0, 0, 0, 1));
    __m256i x4 = _mm256_max_epi32(x2, x3);
    __m128i x5 = _mm256_extracti128_si256(x4, 1);
    __m128i x6 = _mm_max_epi32(_mm256_castsi256_si128(x4), x5);
    return _mm_cvtsi128_si32(x6);
}

static inline float squared_euclidean_distance_v256(const int16_t *lhs,
                                                    const int16_t *rhs,
                                                    size_t size)
{
    const int16_t *last = lhs + size;
    const int16_t *last_aligned = lhs + ((size >> 4) << 4);
    float result = 0.0f;

    static const __m256i iymm_zero = _mm256_setzero_si256();
    __m256 ymm_sum1 = _mm256_setzero_ps();
    __m256 ymm_sum2 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));
            __m256 ymm0 =
                _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(iymm2, iymm_zero));
            __m256 ymm1 =
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(iymm2, iymm_zero));
            ymm_sum1 = _mm256_fmadd_ps(ymm0, ymm0, ymm_sum1);
            ymm_sum2 = _mm256_fmadd_ps(ymm1, ymm1, ymm_sum2);
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));
            __m256 ymm0 =
                _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(iymm2, iymm_zero));
            __m256 ymm1 =
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(iymm2, iymm_zero));
            ymm_sum1 = _mm256_fmadd_ps(ymm0, ymm0, ymm_sum1);
            ymm_sum2 = _mm256_fmadd_ps(ymm1, ymm1, ymm_sum2);
        }
    }
    result = horizontal_add_v256(_mm256_add_ps(ymm_sum1, ymm_sum2));

    float x;
    switch (last - last_aligned) {
    case 15:
        x = lhs[14] - rhs[14];
        result += (x * x);
        /* FALLTHRU */
    case 14:
        x = lhs[13] - rhs[13];
        result += (x * x);
        /* FALLTHRU */
    case 13:
        x = lhs[12] - rhs[12];
        result += (x * x);
        /* FALLTHRU */
    case 12:
        x = lhs[11] - rhs[11];
        result += (x * x);
        /* FALLTHRU */
    case 11:
        x = lhs[10] - rhs[10];
        result += (x * x);
        /* FALLTHRU */
    case 10:
        x = lhs[9] - rhs[9];
        result += (x * x);
        /* FALLTHRU */
    case 9:
        x = lhs[8] - rhs[8];
        result += (x * x);
        /* FALLTHRU */
    case 8:
        x = lhs[7] - rhs[7];
        result += (x * x);
        /* FALLTHRU */
    case 7:
        x = lhs[6] - rhs[6];
        result += (x * x);
        /* FALLTHRU */
    case 6:
        x = lhs[5] - rhs[5];
        result += (x * x);
        /* FALLTHRU */
    case 5:
        x = lhs[4] - rhs[4];
        result += (x * x);
        /* FALLTHRU */
    case 4:
        x = lhs[3] - rhs[3];
        result += (x * x);
        /* FALLTHRU */
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x);
    }
    return result;
}

static inline float weighted_squared_euclidean_distance_v256(const int16_t *lhs,
                                                             const int16_t *rhs,
                                                             const float *wgt,
                                                             size_t size)
{
    const int16_t *last = lhs + size;
    const int16_t *last_aligned = lhs + ((size >> 4) << 4);
    float result = 0.0f;

    static const __m256i iymm_zero = _mm256_setzero_si256();
    __m256 ymm_sum1 = _mm256_setzero_ps();
    __m256 ymm_sum2 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));
            __m256 ymm0 =
                _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(iymm2, iymm_zero));
            __m256 ymm1 =
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(iymm2, iymm_zero));

            ymm_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(ymm0, ymm0),
                                       _mm256_load_ps(wgt), ymm_sum1);
            ymm_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(ymm1, ymm1),
                                       _mm256_load_ps(wgt + 8), ymm_sum2);
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));
            __m256 ymm0 =
                _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(iymm2, iymm_zero));
            __m256 ymm1 =
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(iymm2, iymm_zero));

            ymm_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(ymm0, ymm0),
                                       _mm256_loadu_ps(wgt), ymm_sum1);
            ymm_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(ymm1, ymm1),
                                       _mm256_loadu_ps(wgt + 8), ymm_sum2);
        }
    }
    result = horizontal_add_v256(_mm256_add_ps(ymm_sum1, ymm_sum2));

    float x;
    switch (last - last_aligned) {
    case 15:
        x = lhs[14] - rhs[14];
        result += (x * x * wgt[14]);
        /* FALLTHRU */
    case 14:
        x = lhs[13] - rhs[13];
        result += (x * x * wgt[13]);
        /* FALLTHRU */
    case 13:
        x = lhs[12] - rhs[12];
        result += (x * x * wgt[12]);
        /* FALLTHRU */
    case 12:
        x = lhs[11] - rhs[11];
        result += (x * x * wgt[11]);
        /* FALLTHRU */
    case 11:
        x = lhs[10] - rhs[10];
        result += (x * x * wgt[10]);
        /* FALLTHRU */
    case 10:
        x = lhs[9] - rhs[9];
        result += (x * x * wgt[9]);
        /* FALLTHRU */
    case 9:
        x = lhs[8] - rhs[8];
        result += (x * x * wgt[8]);
        /* FALLTHRU */
    case 8:
        x = lhs[7] - rhs[7];
        result += (x * x * wgt[7]);
        /* FALLTHRU */
    case 7:
        x = lhs[6] - rhs[6];
        result += (x * x * wgt[6]);
        /* FALLTHRU */
    case 6:
        x = lhs[5] - rhs[5];
        result += (x * x * wgt[5]);
        /* FALLTHRU */
    case 5:
        x = lhs[4] - rhs[4];
        result += (x * x * wgt[4]);
        /* FALLTHRU */
    case 4:
        x = lhs[3] - rhs[3];
        result += (x * x * wgt[3]);
        /* FALLTHRU */
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x * wgt[2]);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x * wgt[1]);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x * wgt[0]);
    }
    return result;
}

static inline float manhattan_distance_v256(const int16_t *lhs,
                                            const int16_t *rhs, size_t size)
{
    const int16_t *last = lhs + size;
    const int16_t *last_aligned = lhs + ((size >> 5) << 5);
    float result = 0.0f;

    static const __m256i iymm_zero = _mm256_setzero_si256();
    __m256i iymm_sum1 = _mm256_setzero_si256();
    __m256i iymm_sum2 = _mm256_setzero_si256();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));

            __m256i iymm3 = _mm256_load_si256((const __m256i *)(lhs + 16));
            __m256i iymm4 = _mm256_load_si256((const __m256i *)(rhs + 16));
            __m256i iymm5 = _mm256_sub_epi16(_mm256_max_epi16(iymm3, iymm4),
                                             _mm256_min_epi16(iymm3, iymm4));

            iymm_sum1 = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_unpacklo_epi16(iymm2, iymm_zero),
                                 _mm256_unpackhi_epi16(iymm2, iymm_zero)),
                iymm_sum1);
            iymm_sum2 = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_unpacklo_epi16(iymm5, iymm_zero),
                                 _mm256_unpackhi_epi16(iymm5, iymm_zero)),
                iymm_sum2);
        }

        if ((last - last_aligned) > 15) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));
            iymm_sum1 = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_unpacklo_epi16(iymm2, iymm_zero),
                                 _mm256_unpackhi_epi16(iymm2, iymm_zero)),
                iymm_sum1);
            lhs += 16;
            rhs += 16;
        }
    } else {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));

            __m256i iymm3 = _mm256_loadu_si256((const __m256i *)(lhs + 16));
            __m256i iymm4 = _mm256_loadu_si256((const __m256i *)(rhs + 16));
            __m256i iymm5 = _mm256_sub_epi16(_mm256_max_epi16(iymm3, iymm4),
                                             _mm256_min_epi16(iymm3, iymm4));

            iymm_sum1 = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_unpacklo_epi16(iymm2, iymm_zero),
                                 _mm256_unpackhi_epi16(iymm2, iymm_zero)),
                iymm_sum1);
            iymm_sum2 = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_unpacklo_epi16(iymm5, iymm_zero),
                                 _mm256_unpackhi_epi16(iymm5, iymm_zero)),
                iymm_sum2);
        }

        if ((last - last_aligned) > 15) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));
            iymm_sum1 = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_unpacklo_epi16(iymm2, iymm_zero),
                                 _mm256_unpackhi_epi16(iymm2, iymm_zero)),
                iymm_sum1);
            lhs += 16;
            rhs += 16;
        }
    }
    result = horizontal_add_v256(
        _mm256_cvtepi32_ps(_mm256_add_epi32(iymm_sum1, iymm_sum2)));

    switch (last - lhs) {
    case 15:
        result += fast_abs(lhs[14] - rhs[14]);
        /* FALLTHRU */
    case 14:
        result += fast_abs(lhs[13] - rhs[13]);
        /* FALLTHRU */
    case 13:
        result += fast_abs(lhs[12] - rhs[12]);
        /* FALLTHRU */
    case 12:
        result += fast_abs(lhs[11] - rhs[11]);
        /* FALLTHRU */
    case 11:
        result += fast_abs(lhs[10] - rhs[10]);
        /* FALLTHRU */
    case 10:
        result += fast_abs(lhs[9] - rhs[9]);
        /* FALLTHRU */
    case 9:
        result += fast_abs(lhs[8] - rhs[8]);
        /* FALLTHRU */
    case 8:
        result += fast_abs(lhs[7] - rhs[7]);
        /* FALLTHRU */
    case 7:
        result += fast_abs(lhs[6] - rhs[6]);
        /* FALLTHRU */
    case 6:
        result += fast_abs(lhs[5] - rhs[5]);
        /* FALLTHRU */
    case 5:
        result += fast_abs(lhs[4] - rhs[4]);
        /* FALLTHRU */
    case 4:
        result += fast_abs(lhs[3] - rhs[3]);
        /* FALLTHRU */
    case 3:
        result += fast_abs(lhs[2] - rhs[2]);
        /* FALLTHRU */
    case 2:
        result += fast_abs(lhs[1] - rhs[1]);
        /* FALLTHRU */
    case 1:
        result += fast_abs(lhs[0] - rhs[0]);
    }
    return result;
}

static inline float chebyshev_distance_v256(const int16_t *lhs,
                                            const int16_t *rhs, size_t size)
{
    const int16_t *last = lhs + size;
    const int16_t *last_aligned = lhs + ((size >> 5) << 5);
    int32_t result = 0;

    static const __m256i iymm_zero = _mm256_setzero_si256();
    __m256i iymm_max = _mm256_setzero_si256();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));

            __m256i iymm3 = _mm256_load_si256((const __m256i *)(lhs + 16));
            __m256i iymm4 = _mm256_load_si256((const __m256i *)(rhs + 16));
            __m256i iymm5 = _mm256_sub_epi16(_mm256_max_epi16(iymm3, iymm4),
                                             _mm256_min_epi16(iymm3, iymm4));

            iymm_max =
                _mm256_max_epu16(_mm256_max_epu16(iymm2, iymm5), iymm_max);
        }

        if ((last - last_aligned) > 15) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));
            iymm_max = _mm256_max_epu16(iymm2, iymm_max);
            lhs += 16;
            rhs += 16;
        }
    } else {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));

            __m256i iymm3 = _mm256_loadu_si256((const __m256i *)(lhs + 16));
            __m256i iymm4 = _mm256_loadu_si256((const __m256i *)(rhs + 16));
            __m256i iymm5 = _mm256_sub_epi16(_mm256_max_epi16(iymm3, iymm4),
                                             _mm256_min_epi16(iymm3, iymm4));

            iymm_max =
                _mm256_max_epu16(_mm256_max_epu16(iymm2, iymm5), iymm_max);
        }

        if ((last - last_aligned) > 15) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi16(_mm256_max_epi16(iymm0, iymm1),
                                             _mm256_min_epi16(iymm0, iymm1));
            iymm_max = _mm256_max_epu16(iymm2, iymm_max);
            lhs += 16;
            rhs += 16;
        }
    }
    result = horizontal_max_v256(
        _mm256_max_epi32(_mm256_unpacklo_epi16(iymm_max, iymm_zero),
                         _mm256_unpackhi_epi16(iymm_max, iymm_zero)));

    int32_t x;
    switch (last - lhs) {
    case 15:
        x = fast_abs(lhs[14] - rhs[14]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 14:
        x = fast_abs(lhs[13] - rhs[13]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 13:
        x = fast_abs(lhs[12] - rhs[12]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 12:
        x = fast_abs(lhs[11] - rhs[11]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 11:
        x = fast_abs(lhs[10] - rhs[10]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 10:
        x = fast_abs(lhs[9] - rhs[9]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 9:
        x = fast_abs(lhs[8] - rhs[8]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 8:
        x = fast_abs(lhs[7] - rhs[7]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 7:
        x = fast_abs(lhs[6] - rhs[6]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 6:
        x = fast_abs(lhs[5] - rhs[5]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 5:
        x = fast_abs(lhs[4] - rhs[4]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 4:
        x = fast_abs(lhs[3] - rhs[3]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 3:
        x = fast_abs(lhs[2] - rhs[2]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 2:
        x = fast_abs(lhs[1] - rhs[1]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 1:
        x = fast_abs(lhs[0] - rhs[0]);
        if (result < x) {
            result = x;
        }
    }
    return result;
}

static inline float inner_product_v256(const int16_t *lhs, const int16_t *rhs,
                                       size_t size)
{
    const int16_t *last = lhs + size;
    const int16_t *last_aligned = lhs + ((size >> 5) << 5);
    float sum = 0.0f;

    __m256 ymm_sum1 = _mm256_setzero_ps();
    __m256 ymm_sum2 = _mm256_setzero_ps();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m256i ymm0 =
                _mm256_madd_epi16(_mm256_load_si256((const __m256i *)lhs),
                                  _mm256_load_si256((const __m256i *)rhs));
            __m256i ymm1 = _mm256_madd_epi16(
                _mm256_load_si256((const __m256i *)(lhs + 16)),
                _mm256_load_si256((const __m256i *)(rhs + 16)));

            ymm_sum1 = _mm256_add_ps(_mm256_cvtepi32_ps(ymm0), ymm_sum1);
            ymm_sum2 = _mm256_add_ps(_mm256_cvtepi32_ps(ymm1), ymm_sum2);
        }

        if ((last - last_aligned) > 15) {
            ymm_sum1 =
                _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(
                                  _mm256_load_si256((const __m256i *)lhs),
                                  _mm256_load_si256((const __m256i *)rhs))),
                              ymm_sum1);
            lhs += 16;
            rhs += 16;
        }
    } else {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m256i ymm0 =
                _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)lhs),
                                  _mm256_loadu_si256((const __m256i *)rhs));
            __m256i ymm1 = _mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i *)(lhs + 16)),
                _mm256_loadu_si256((const __m256i *)(rhs + 16)));

            ymm_sum1 = _mm256_add_ps(_mm256_cvtepi32_ps(ymm0), ymm_sum1);
            ymm_sum2 = _mm256_add_ps(_mm256_cvtepi32_ps(ymm1), ymm_sum2);
        }

        if ((last - last_aligned) > 15) {
            ymm_sum1 =
                _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(
                                  _mm256_loadu_si256((const __m256i *)lhs),
                                  _mm256_loadu_si256((const __m256i *)rhs))),
                              ymm_sum1);
            lhs += 16;
            rhs += 16;
        }
    }
    sum = horizontal_add_v256(_mm256_add_ps(ymm_sum1, ymm_sum2));

    switch (last - lhs) {
    case 15:
        sum += (lhs[14] * rhs[14]);
        /* FALLTHRU */
    case 14:
        sum += (lhs[13] * rhs[13]);
        /* FALLTHRU */
    case 13:
        sum += (lhs[12] * rhs[12]);
        /* FALLTHRU */
    case 12:
        sum += (lhs[11] * rhs[11]);
        /* FALLTHRU */
    case 11:
        sum += (lhs[10] * rhs[10]);
        /* FALLTHRU */
    case 10:
        sum += (lhs[9] * rhs[9]);
        /* FALLTHRU */
    case 9:
        sum += (lhs[8] * rhs[8]);
        /* FALLTHRU */
    case 8:
        sum += (lhs[7] * rhs[7]);
        /* FALLTHRU */
    case 7:
        sum += (lhs[6] * rhs[6]);
        /* FALLTHRU */
    case 6:
        sum += (lhs[5] * rhs[5]);
        /* FALLTHRU */
    case 5:
        sum += (lhs[4] * rhs[4]);
        /* FALLTHRU */
    case 4:
        sum += (lhs[3] * rhs[3]);
        /* FALLTHRU */
    case 3:
        sum += (lhs[2] * rhs[2]);
        /* FALLTHRU */
    case 2:
        sum += (lhs[1] * rhs[1]);
        /* FALLTHRU */
    case 1:
        sum += (lhs[0] * rhs[0]);
    }
    return sum;
}

static inline float squared_euclidean_distance_v256(const int8_t *lhs,
                                                    const int8_t *rhs,
                                                    size_t size)
{
    const int8_t *last = lhs + size;
    const int8_t *last_aligned = lhs + ((size >> 6) << 6);
    float result = 0.0;

    static const __m256i iymm_zero = _mm256_setzero_si256();
    __m256i iymm_sum1 = _mm256_setzero_si256();
    __m256i iymm_sum2 = _mm256_setzero_si256();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 64, rhs += 64) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));

            __m256i iymm3 = _mm256_load_si256((const __m256i *)(lhs + 32));
            __m256i iymm4 = _mm256_load_si256((const __m256i *)(rhs + 32));
            __m256i iymm5 = _mm256_sub_epi8(_mm256_max_epi8(iymm3, iymm4),
                                            _mm256_min_epi8(iymm3, iymm4));

            iymm0 = _mm256_unpacklo_epi8(iymm2, iymm_zero);
            iymm1 = _mm256_unpackhi_epi8(iymm2, iymm_zero);
            iymm3 = _mm256_unpacklo_epi8(iymm5, iymm_zero);
            iymm4 = _mm256_unpackhi_epi8(iymm5, iymm_zero);
            iymm2 = _mm256_add_epi32(_mm256_madd_epi16(iymm0, iymm0),
                                     _mm256_madd_epi16(iymm1, iymm1));
            iymm5 = _mm256_add_epi32(_mm256_madd_epi16(iymm3, iymm3),
                                     _mm256_madd_epi16(iymm4, iymm4));

            iymm_sum1 = _mm256_add_epi32(iymm2, iymm_sum1);
            iymm_sum2 = _mm256_add_epi32(iymm5, iymm_sum2);
        }

        if ((last - last_aligned) > 31) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));
            iymm0 = _mm256_unpacklo_epi8(iymm2, iymm_zero);
            iymm1 = _mm256_unpackhi_epi8(iymm2, iymm_zero);
            iymm2 = _mm256_add_epi32(_mm256_madd_epi16(iymm0, iymm0),
                                     _mm256_madd_epi16(iymm1, iymm1));

            iymm_sum1 = _mm256_add_epi32(iymm2, iymm_sum1);
            lhs += 32;
            rhs += 32;
        }
    } else {
        for (; lhs != last_aligned; lhs += 64, rhs += 64) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));

            __m256i iymm3 = _mm256_loadu_si256((const __m256i *)(lhs + 32));
            __m256i iymm4 = _mm256_loadu_si256((const __m256i *)(rhs + 32));
            __m256i iymm5 = _mm256_sub_epi8(_mm256_max_epi8(iymm3, iymm4),
                                            _mm256_min_epi8(iymm3, iymm4));

            iymm0 = _mm256_unpacklo_epi8(iymm2, iymm_zero);
            iymm1 = _mm256_unpackhi_epi8(iymm2, iymm_zero);
            iymm3 = _mm256_unpacklo_epi8(iymm5, iymm_zero);
            iymm4 = _mm256_unpackhi_epi8(iymm5, iymm_zero);
            iymm2 = _mm256_add_epi32(_mm256_madd_epi16(iymm0, iymm0),
                                     _mm256_madd_epi16(iymm1, iymm1));
            iymm5 = _mm256_add_epi32(_mm256_madd_epi16(iymm3, iymm3),
                                     _mm256_madd_epi16(iymm4, iymm4));

            iymm_sum1 = _mm256_add_epi32(iymm2, iymm_sum1);
            iymm_sum2 = _mm256_add_epi32(iymm5, iymm_sum2);
        }

        if ((last - last_aligned) > 31) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));
            iymm0 = _mm256_unpacklo_epi8(iymm2, iymm_zero);
            iymm1 = _mm256_unpackhi_epi8(iymm2, iymm_zero);
            iymm2 = _mm256_add_epi32(_mm256_madd_epi16(iymm0, iymm0),
                                     _mm256_madd_epi16(iymm1, iymm1));

            iymm_sum1 = _mm256_add_epi32(iymm2, iymm_sum1);
            lhs += 32;
            rhs += 32;
        }
    }
    result = horizontal_add_v256(
        _mm256_cvtepi32_ps(_mm256_add_epi32(iymm_sum1, iymm_sum2)));

    int32_t x;
    switch (last - lhs) {
    case 31:
        x = lhs[30] - rhs[30];
        result += (x * x);
        /* FALLTHRU */
    case 30:
        x = lhs[29] - rhs[29];
        result += (x * x);
        /* FALLTHRU */
    case 29:
        x = lhs[28] - rhs[28];
        result += (x * x);
        /* FALLTHRU */
    case 28:
        x = lhs[27] - rhs[27];
        result += (x * x);
        /* FALLTHRU */
    case 27:
        x = lhs[26] - rhs[26];
        result += (x * x);
        /* FALLTHRU */
    case 26:
        x = lhs[25] - rhs[25];
        result += (x * x);
        /* FALLTHRU */
    case 25:
        x = lhs[24] - rhs[24];
        result += (x * x);
        /* FALLTHRU */
    case 24:
        x = lhs[23] - rhs[23];
        result += (x * x);
        /* FALLTHRU */
    case 23:
        x = lhs[22] - rhs[22];
        result += (x * x);
        /* FALLTHRU */
    case 22:
        x = lhs[21] - rhs[21];
        result += (x * x);
        /* FALLTHRU */
    case 21:
        x = lhs[20] - rhs[20];
        result += (x * x);
        /* FALLTHRU */
    case 20:
        x = lhs[19] - rhs[19];
        result += (x * x);
        /* FALLTHRU */
    case 19:
        x = lhs[18] - rhs[18];
        result += (x * x);
        /* FALLTHRU */
    case 18:
        x = lhs[17] - rhs[17];
        result += (x * x);
        /* FALLTHRU */
    case 17:
        x = lhs[16] - rhs[16];
        result += (x * x);
        /* FALLTHRU */
    case 16:
        x = lhs[15] - rhs[15];
        result += (x * x);
        /* FALLTHRU */
    case 15:
        x = lhs[14] - rhs[14];
        result += (x * x);
        /* FALLTHRU */
    case 14:
        x = lhs[13] - rhs[13];
        result += (x * x);
        /* FALLTHRU */
    case 13:
        x = lhs[12] - rhs[12];
        result += (x * x);
        /* FALLTHRU */
    case 12:
        x = lhs[11] - rhs[11];
        result += (x * x);
        /* FALLTHRU */
    case 11:
        x = lhs[10] - rhs[10];
        result += (x * x);
        /* FALLTHRU */
    case 10:
        x = lhs[9] - rhs[9];
        result += (x * x);
        /* FALLTHRU */
    case 9:
        x = lhs[8] - rhs[8];
        result += (x * x);
        /* FALLTHRU */
    case 8:
        x = lhs[7] - rhs[7];
        result += (x * x);
        /* FALLTHRU */
    case 7:
        x = lhs[6] - rhs[6];
        result += (x * x);
        /* FALLTHRU */
    case 6:
        x = lhs[5] - rhs[5];
        result += (x * x);
        /* FALLTHRU */
    case 5:
        x = lhs[4] - rhs[4];
        result += (x * x);
        /* FALLTHRU */
    case 4:
        x = lhs[3] - rhs[3];
        result += (x * x);
        /* FALLTHRU */
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x);
    }
    return result;
}

static inline float manhattan_distance_v256(const int8_t *lhs,
                                            const int8_t *rhs, size_t size)
{
    const int8_t *last = lhs + size;
    const int8_t *last_aligned = lhs + ((size >> 6) << 6);
    int32_t result = 0;

    static const __m256i iymm_zero = _mm256_setzero_si256();
    __m256i iymm_sum1 = _mm256_setzero_si256();
    __m256i iymm_sum2 = _mm256_setzero_si256();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 64, rhs += 64) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));

            __m256i iymm3 = _mm256_load_si256((const __m256i *)(lhs + 32));
            __m256i iymm4 = _mm256_load_si256((const __m256i *)(rhs + 32));
            __m256i iymm5 = _mm256_sub_epi8(_mm256_max_epi8(iymm3, iymm4),
                                            _mm256_min_epi8(iymm3, iymm4));

            iymm0 = _mm256_add_epi16(_mm256_unpacklo_epi8(iymm2, iymm_zero),
                                     _mm256_unpackhi_epi8(iymm2, iymm_zero));
            iymm1 = _mm256_add_epi16(_mm256_unpacklo_epi8(iymm5, iymm_zero),
                                     _mm256_unpackhi_epi8(iymm5, iymm_zero));
            iymm2 = _mm256_add_epi16(iymm0, iymm1);

            iymm_sum1 = _mm256_add_epi32(
                _mm256_unpacklo_epi16(iymm2, iymm_zero), iymm_sum1);
            iymm_sum2 = _mm256_add_epi32(
                _mm256_unpackhi_epi16(iymm2, iymm_zero), iymm_sum2);
        }

        if ((last - last_aligned) > 31) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));
            __m256i iymm3 =
                _mm256_add_epi16(_mm256_unpacklo_epi8(iymm2, iymm_zero),
                                 _mm256_unpackhi_epi8(iymm2, iymm_zero));
            iymm_sum1 = _mm256_add_epi32(
                _mm256_unpacklo_epi16(iymm3, iymm_zero), iymm_sum1);
            iymm_sum2 = _mm256_add_epi32(
                _mm256_unpackhi_epi16(iymm3, iymm_zero), iymm_sum2);
            lhs += 32;
            rhs += 32;
        }
    } else {
        for (; lhs != last_aligned; lhs += 64, rhs += 64) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));

            __m256i iymm3 = _mm256_loadu_si256((const __m256i *)(lhs + 32));
            __m256i iymm4 = _mm256_loadu_si256((const __m256i *)(rhs + 32));
            __m256i iymm5 = _mm256_sub_epi8(_mm256_max_epi8(iymm3, iymm4),
                                            _mm256_min_epi8(iymm3, iymm4));

            iymm0 = _mm256_add_epi16(_mm256_unpacklo_epi8(iymm2, iymm_zero),
                                     _mm256_unpackhi_epi8(iymm2, iymm_zero));
            iymm1 = _mm256_add_epi16(_mm256_unpacklo_epi8(iymm5, iymm_zero),
                                     _mm256_unpackhi_epi8(iymm5, iymm_zero));
            iymm2 = _mm256_add_epi16(iymm0, iymm1);

            iymm_sum1 = _mm256_add_epi32(
                _mm256_unpacklo_epi16(iymm2, iymm_zero), iymm_sum1);
            iymm_sum2 = _mm256_add_epi32(
                _mm256_unpackhi_epi16(iymm2, iymm_zero), iymm_sum2);
        }

        if ((last - last_aligned) > 31) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));
            __m256i iymm3 =
                _mm256_add_epi16(_mm256_unpacklo_epi8(iymm2, iymm_zero),
                                 _mm256_unpackhi_epi8(iymm2, iymm_zero));
            iymm_sum1 = _mm256_add_epi32(
                _mm256_unpacklo_epi16(iymm3, iymm_zero), iymm_sum1);
            iymm_sum2 = _mm256_add_epi32(
                _mm256_unpackhi_epi16(iymm3, iymm_zero), iymm_sum2);
            lhs += 32;
            rhs += 32;
        }
    }
    result = horizontal_add_v256(_mm256_add_epi32(iymm_sum1, iymm_sum2));

    switch (last - lhs) {
    case 31:
        result += fast_abs(lhs[30] - rhs[30]);
        /* FALLTHRU */
    case 30:
        result += fast_abs(lhs[29] - rhs[29]);
        /* FALLTHRU */
    case 29:
        result += fast_abs(lhs[28] - rhs[28]);
        /* FALLTHRU */
    case 28:
        result += fast_abs(lhs[27] - rhs[27]);
        /* FALLTHRU */
    case 27:
        result += fast_abs(lhs[26] - rhs[26]);
        /* FALLTHRU */
    case 26:
        result += fast_abs(lhs[25] - rhs[25]);
        /* FALLTHRU */
    case 25:
        result += fast_abs(lhs[24] - rhs[24]);
        /* FALLTHRU */
    case 24:
        result += fast_abs(lhs[23] - rhs[23]);
        /* FALLTHRU */
    case 23:
        result += fast_abs(lhs[22] - rhs[22]);
        /* FALLTHRU */
    case 22:
        result += fast_abs(lhs[21] - rhs[21]);
        /* FALLTHRU */
    case 21:
        result += fast_abs(lhs[20] - rhs[20]);
        /* FALLTHRU */
    case 20:
        result += fast_abs(lhs[19] - rhs[19]);
        /* FALLTHRU */
    case 19:
        result += fast_abs(lhs[18] - rhs[18]);
        /* FALLTHRU */
    case 18:
        result += fast_abs(lhs[17] - rhs[17]);
        /* FALLTHRU */
    case 17:
        result += fast_abs(lhs[16] - rhs[16]);
        /* FALLTHRU */
    case 16:
        result += fast_abs(lhs[15] - rhs[15]);
        /* FALLTHRU */
    case 15:
        result += fast_abs(lhs[14] - rhs[14]);
        /* FALLTHRU */
    case 14:
        result += fast_abs(lhs[13] - rhs[13]);
        /* FALLTHRU */
    case 13:
        result += fast_abs(lhs[12] - rhs[12]);
        /* FALLTHRU */
    case 12:
        result += fast_abs(lhs[11] - rhs[11]);
        /* FALLTHRU */
    case 11:
        result += fast_abs(lhs[10] - rhs[10]);
        /* FALLTHRU */
    case 10:
        result += fast_abs(lhs[9] - rhs[9]);
        /* FALLTHRU */
    case 9:
        result += fast_abs(lhs[8] - rhs[8]);
        /* FALLTHRU */
    case 8:
        result += fast_abs(lhs[7] - rhs[7]);
        /* FALLTHRU */
    case 7:
        result += fast_abs(lhs[6] - rhs[6]);
        /* FALLTHRU */
    case 6:
        result += fast_abs(lhs[5] - rhs[5]);
        /* FALLTHRU */
    case 5:
        result += fast_abs(lhs[4] - rhs[4]);
        /* FALLTHRU */
    case 4:
        result += fast_abs(lhs[3] - rhs[3]);
        /* FALLTHRU */
    case 3:
        result += fast_abs(lhs[2] - rhs[2]);
        /* FALLTHRU */
    case 2:
        result += fast_abs(lhs[1] - rhs[1]);
        /* FALLTHRU */
    case 1:
        result += fast_abs(lhs[0] - rhs[0]);
    }
    return result;
}

static inline float chebyshev_distance_v256(const int8_t *lhs,
                                            const int8_t *rhs, size_t size)
{
    const int8_t *last = lhs + size;
    const int8_t *last_aligned = lhs + ((size >> 6) << 6);
    int32_t result = 0;

    static const __m256i iymm_zero = _mm256_setzero_si256();
    __m256i iymm_max = _mm256_setzero_si256();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 64, rhs += 64) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));

            __m256i iymm3 = _mm256_load_si256((const __m256i *)(lhs + 32));
            __m256i iymm4 = _mm256_load_si256((const __m256i *)(rhs + 32));
            __m256i iymm5 = _mm256_sub_epi8(_mm256_max_epi8(iymm3, iymm4),
                                            _mm256_min_epi8(iymm3, iymm4));

            iymm_max = _mm256_max_epu8(_mm256_max_epu8(iymm2, iymm5), iymm_max);
        }

        if ((last - last_aligned) > 31) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));
            iymm_max = _mm256_max_epu8(iymm2, iymm_max);
            lhs += 32;
            rhs += 32;
        }
    } else {
        for (; lhs != last_aligned; lhs += 64, rhs += 64) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));

            __m256i iymm3 = _mm256_loadu_si256((const __m256i *)(lhs + 32));
            __m256i iymm4 = _mm256_loadu_si256((const __m256i *)(rhs + 32));
            __m256i iymm5 = _mm256_sub_epi8(_mm256_max_epi8(iymm3, iymm4),
                                            _mm256_min_epi8(iymm3, iymm4));

            iymm_max = _mm256_max_epu8(_mm256_max_epu8(iymm2, iymm5), iymm_max);
        }

        if ((last - last_aligned) > 31) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_sub_epi8(_mm256_max_epi8(iymm0, iymm1),
                                            _mm256_min_epi8(iymm0, iymm1));
            iymm_max = _mm256_max_epu8(iymm2, iymm_max);
            lhs += 32;
            rhs += 32;
        }
    }
    iymm_max = _mm256_max_epi16(_mm256_unpacklo_epi8(iymm_max, iymm_zero),
                                _mm256_unpackhi_epi8(iymm_max, iymm_zero));
    result = horizontal_max_v256(
        _mm256_max_epi32(_mm256_unpacklo_epi16(iymm_max, iymm_zero),
                         _mm256_unpackhi_epi16(iymm_max, iymm_zero)));

    int32_t x;
    switch (last - lhs) {
    case 31:
        x = fast_abs(lhs[30] - rhs[30]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 30:
        x = fast_abs(lhs[29] - rhs[29]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 29:
        x = fast_abs(lhs[28] - rhs[28]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 28:
        x = fast_abs(lhs[27] - rhs[27]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 27:
        x = fast_abs(lhs[26] - rhs[26]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 26:
        x = fast_abs(lhs[25] - rhs[25]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 25:
        x = fast_abs(lhs[24] - rhs[24]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 24:
        x = fast_abs(lhs[23] - rhs[23]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 23:
        x = fast_abs(lhs[22] - rhs[22]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 22:
        x = fast_abs(lhs[21] - rhs[21]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 21:
        x = fast_abs(lhs[20] - rhs[20]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 20:
        x = fast_abs(lhs[19] - rhs[19]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 19:
        x = fast_abs(lhs[18] - rhs[18]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 18:
        x = fast_abs(lhs[17] - rhs[17]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 17:
        x = fast_abs(lhs[16] - rhs[16]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 16:
        x = fast_abs(lhs[15] - rhs[15]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 15:
        x = fast_abs(lhs[14] - rhs[14]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 14:
        x = fast_abs(lhs[13] - rhs[13]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 13:
        x = fast_abs(lhs[12] - rhs[12]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 12:
        x = fast_abs(lhs[11] - rhs[11]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 11:
        x = fast_abs(lhs[10] - rhs[10]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 10:
        x = fast_abs(lhs[9] - rhs[9]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 9:
        x = fast_abs(lhs[8] - rhs[8]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 8:
        x = fast_abs(lhs[7] - rhs[7]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 7:
        x = fast_abs(lhs[6] - rhs[6]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 6:
        x = fast_abs(lhs[5] - rhs[5]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 5:
        x = fast_abs(lhs[4] - rhs[4]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 4:
        x = fast_abs(lhs[3] - rhs[3]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 3:
        x = fast_abs(lhs[2] - rhs[2]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 2:
        x = fast_abs(lhs[1] - rhs[1]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 1:
        x = fast_abs(lhs[0] - rhs[0]);
        if (result < x) {
            result = x;
        }
    }
    return result;
}

static inline float inner_product_v256(const int8_t *lhs, const int8_t *rhs,
                                       size_t size)
{
    const int8_t *last = lhs + size;
    const int8_t *last_aligned = lhs + ((size >> 6) << 6);
    int32_t sum = 0;

    __m256i iymm_sum1 = _mm256_setzero_si256();
    __m256i iymm_sum2 = _mm256_setzero_si256();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 64, rhs += 64) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm0)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm1)));
            __m256i iymm3 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm0, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm1, 1)));

            __m256i iymm4 = _mm256_load_si256((const __m256i *)(lhs + 32));
            __m256i iymm5 = _mm256_load_si256((const __m256i *)(rhs + 32));
            iymm0 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm4)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm5)));
            iymm1 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm4, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm5, 1)));

            iymm_sum1 =
                _mm256_add_epi32(_mm256_add_epi32(iymm0, iymm1), iymm_sum1);
            iymm_sum2 =
                _mm256_add_epi32(_mm256_add_epi32(iymm2, iymm3), iymm_sum2);
        }

        if ((last - last_aligned) > 31) {
            __m256i iymm0 = _mm256_load_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_load_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm0)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm1)));
            __m256i iymm3 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm0, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm1, 1)));

            iymm_sum2 =
                _mm256_add_epi32(_mm256_add_epi32(iymm2, iymm3), iymm_sum2);
            lhs += 32;
            rhs += 32;
        }
    } else {
        for (; lhs != last_aligned; lhs += 64, rhs += 64) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm0)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm1)));
            __m256i iymm3 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm0, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm1, 1)));

            __m256i iymm4 = _mm256_loadu_si256((const __m256i *)(lhs + 32));
            __m256i iymm5 = _mm256_loadu_si256((const __m256i *)(rhs + 32));
            iymm0 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm4)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm5)));
            iymm1 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm4, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm5, 1)));

            iymm_sum1 =
                _mm256_add_epi32(_mm256_add_epi32(iymm0, iymm1), iymm_sum1);
            iymm_sum2 =
                _mm256_add_epi32(_mm256_add_epi32(iymm2, iymm3), iymm_sum2);
        }

        if ((last - last_aligned) > 31) {
            __m256i iymm0 = _mm256_loadu_si256((const __m256i *)lhs);
            __m256i iymm1 = _mm256_loadu_si256((const __m256i *)rhs);
            __m256i iymm2 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm0)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(iymm1)));
            __m256i iymm3 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm0, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(iymm1, 1)));

            iymm_sum2 =
                _mm256_add_epi32(_mm256_add_epi32(iymm2, iymm3), iymm_sum2);
            lhs += 32;
            rhs += 32;
        }
    }
    sum = horizontal_add_v256(_mm256_add_epi32(iymm_sum1, iymm_sum2));

    switch (last - lhs) {
    case 31:
        sum += (lhs[30] * rhs[30]);
        /* FALLTHRU */
    case 30:
        sum += (lhs[29] * rhs[29]);
        /* FALLTHRU */
    case 29:
        sum += (lhs[28] * rhs[28]);
        /* FALLTHRU */
    case 28:
        sum += (lhs[27] * rhs[27]);
        /* FALLTHRU */
    case 27:
        sum += (lhs[26] * rhs[26]);
        /* FALLTHRU */
    case 26:
        sum += (lhs[25] * rhs[25]);
        /* FALLTHRU */
    case 25:
        sum += (lhs[24] * rhs[24]);
        /* FALLTHRU */
    case 24:
        sum += (lhs[23] * rhs[23]);
        /* FALLTHRU */
    case 23:
        sum += (lhs[22] * rhs[22]);
        /* FALLTHRU */
    case 22:
        sum += (lhs[21] * rhs[21]);
        /* FALLTHRU */
    case 21:
        sum += (lhs[20] * rhs[20]);
        /* FALLTHRU */
    case 20:
        sum += (lhs[19] * rhs[19]);
        /* FALLTHRU */
    case 19:
        sum += (lhs[18] * rhs[18]);
        /* FALLTHRU */
    case 18:
        sum += (lhs[17] * rhs[17]);
        /* FALLTHRU */
    case 17:
        sum += (lhs[16] * rhs[16]);
        /* FALLTHRU */
    case 16:
        sum += (lhs[15] * rhs[15]);
        /* FALLTHRU */
    case 15:
        sum += (lhs[14] * rhs[14]);
        /* FALLTHRU */
    case 14:
        sum += (lhs[13] * rhs[13]);
        /* FALLTHRU */
    case 13:
        sum += (lhs[12] * rhs[12]);
        /* FALLTHRU */
    case 12:
        sum += (lhs[11] * rhs[11]);
        /* FALLTHRU */
    case 11:
        sum += (lhs[10] * rhs[10]);
        /* FALLTHRU */
    case 10:
        sum += (lhs[9] * rhs[9]);
        /* FALLTHRU */
    case 9:
        sum += (lhs[8] * rhs[8]);
        /* FALLTHRU */
    case 8:
        sum += (lhs[7] * rhs[7]);
        /* FALLTHRU */
    case 7:
        sum += (lhs[6] * rhs[6]);
        /* FALLTHRU */
    case 6:
        sum += (lhs[5] * rhs[5]);
        /* FALLTHRU */
    case 5:
        sum += (lhs[4] * rhs[4]);
        /* FALLTHRU */
    case 4:
        sum += (lhs[3] * rhs[3]);
        /* FALLTHRU */
    case 3:
        sum += (lhs[2] * rhs[2]);
        /* FALLTHRU */
    case 2:
        sum += (lhs[1] * rhs[1]);
        /* FALLTHRU */
    case 1:
        sum += (lhs[0] * rhs[0]);
    }
    return sum;
}
#endif // __AVX2__

#if defined(__AVX512F__)
static inline float horizontal_add_v512(__m512 v)
{
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high =
        _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));

    return horizontal_add_v256(low) + horizontal_add_v256(high);
}

static inline float
squared_euclidean_distance_v512(const float *lhs, const float *rhs, size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 5) << 5);
    float result = 0.0f;

    __m512 zmm_sum1 = _mm512_setzero_ps();
    __m512 zmm_sum2 = _mm512_setzero_ps();

    if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m512 zmm0 =
                _mm512_sub_ps(_mm512_load_ps(lhs), _mm512_load_ps(rhs));
            __m512 zmm1 = _mm512_sub_ps(_mm512_load_ps(lhs + 16),
                                        _mm512_load_ps(rhs + 16));
            zmm_sum1 = _mm512_fmadd_ps(zmm0, zmm0, zmm_sum1);
            zmm_sum2 = _mm512_fmadd_ps(zmm1, zmm1, zmm_sum2);
        }

        if ((last - last_aligned) > 15) {
            __m512 zmm0 =
                _mm512_sub_ps(_mm512_load_ps(lhs), _mm512_load_ps(rhs));
            zmm_sum1 = _mm512_fmadd_ps(zmm0, zmm0, zmm_sum1);
            lhs += 16;
            rhs += 16;
        }
    } else {
        for (; lhs != last_aligned; lhs += 32, rhs += 32) {
            __m512 zmm0 =
                _mm512_sub_ps(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs));
            __m512 zmm1 = _mm512_sub_ps(_mm512_loadu_ps(lhs + 16),
                                        _mm512_loadu_ps(rhs + 16));
            zmm_sum1 = _mm512_fmadd_ps(zmm0, zmm0, zmm_sum1);
            zmm_sum2 = _mm512_fmadd_ps(zmm1, zmm1, zmm_sum2);
        }

        if ((last - last_aligned) > 15) {
            __m512 zmm0 =
                _mm512_sub_ps(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs));
            zmm_sum1 = _mm512_fmadd_ps(zmm0, zmm0, zmm_sum1);
            lhs += 16;
            rhs += 16;
        }
    }
    result = horizontal_add_v512(_mm512_add_ps(zmm_sum1, zmm_sum2));

    float x;
    switch (last - lhs) {
    case 15:
        x = lhs[14] - rhs[14];
        result += (x * x);
        /* FALLTHRU */
    case 14:
        x = lhs[13] - rhs[13];
        result += (x * x);
        /* FALLTHRU */
    case 13:
        x = lhs[12] - rhs[12];
        result += (x * x);
        /* FALLTHRU */
    case 12:
        x = lhs[11] - rhs[11];
        result += (x * x);
        /* FALLTHRU */
    case 11:
        x = lhs[10] - rhs[10];
        result += (x * x);
        /* FALLTHRU */
    case 10:
        x = lhs[9] - rhs[9];
        result += (x * x);
        /* FALLTHRU */
    case 9:
        x = lhs[8] - rhs[8];
        result += (x * x);
        /* FALLTHRU */
    case 8:
        x = lhs[7] - rhs[7];
        result += (x * x);
        /* FALLTHRU */
    case 7:
        x = lhs[6] - rhs[6];
        result += (x * x);
        /* FALLTHRU */
    case 6:
        x = lhs[5] - rhs[5];
        result += (x * x);
        /* FALLTHRU */
    case 5:
        x = lhs[4] - rhs[4];
        result += (x * x);
        /* FALLTHRU */
    case 4:
        x = lhs[3] - rhs[3];
        result += (x * x);
        /* FALLTHRU */
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x);
    }
    return result;
}
#endif // __AVX512F__

template <typename T>
static inline float horizontal_mean(const T *lhs, size_t size)
{
    const T *last = lhs + size;
    const T *last_aligned = lhs + ((size >> 2) << 2);
    float sum = 0.0f;

    for (; lhs != last_aligned; lhs += 4) {
        sum += (lhs[3] + lhs[2] + lhs[1] + lhs[0]);
    }
    switch (last - last_aligned) {
    case 3:
        sum += lhs[2];
        /* FALLTHRU */
    case 2:
        sum += lhs[1];
        /* FALLTHRU */
    case 1:
        sum += lhs[0];
    }
    return (sum / (float)size);
}

template <typename T>
static inline float squared_euclidean_distance(const T *lhs, const T *rhs,
                                               size_t size)
{
    const T *last = lhs + size;
    const T *last_aligned = lhs + ((size >> 2) << 2);
    float result = 0.0f;

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        float x3 = lhs[3] - rhs[3];
        float x2 = lhs[2] - rhs[2];
        float x1 = lhs[1] - rhs[1];
        float x0 = lhs[0] - rhs[0];
        result += (float)(x3 * x3) + (float)(x2 * x2) + (float)(x1 * x1) +
                  (float)(x0 * x0);
    }

    float x;
    switch (last - last_aligned) {
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x);
    }
    return result;
}

template <typename T>
static inline float
normalized_squared_euclidean_distance(const T *lhs, const T *rhs, size_t size)
{
    const T *last = lhs + size;
    const T *last_aligned = lhs + ((size >> 2) << 2);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float avg1 = horizontal_mean(lhs, size);
    float avg2 = horizontal_mean(rhs, size);

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        float x31 = lhs[3] - avg1;
        float x32 = rhs[3] - avg2;
        float x21 = lhs[2] - avg1;
        float x22 = rhs[2] - avg2;
        float x11 = lhs[1] - avg1;
        float x12 = rhs[1] - avg2;
        float x01 = lhs[0] - avg1;
        float x02 = rhs[0] - avg2;
        sum2 += (x31 * x31 + x32 * x32 + x21 * x21 + x22 * x22 + x11 * x11 +
                 x12 * x12 + x01 * x01 + x02 * x02);

        float x33 = x31 - x32;
        float x23 = x21 - x22;
        float x13 = x11 - x12;
        float x03 = x01 - x02;
        sum1 += (x33 * x33 + x23 * x23 + x13 * x13 + x03 * x03);
    }

    float x1, x2, x3;
    switch (last - last_aligned) {
    case 3:
        x1 = lhs[2] - avg1;
        x2 = rhs[2] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1] - avg1;
        x2 = rhs[1] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0] - avg1;
        x2 = rhs[0] - avg2;
        x3 = x1 - x2;
        sum1 += (x3 * x3);
        sum2 += (x1 * x1 + x2 * x2);
    }
    return (sum1 / sum2 / 2.0f);
}

template <typename T>
static inline float
weighted_squared_euclidean_distance(const T *lhs, const T *rhs,
                                    const float *wgt, size_t size)
{
    const T *last = lhs + size;
    const T *last_aligned = lhs + ((size >> 2) << 2);
    float result = 0.0f;

    for (; lhs != last_aligned; lhs += 4, rhs += 4, wgt += 4) {
        float x3 = lhs[3] - rhs[3];
        float x2 = lhs[2] - rhs[2];
        float x1 = lhs[1] - rhs[1];
        float x0 = lhs[0] - rhs[0];
        result += (x3 * x3 * wgt[3]) + (x2 * x2 * wgt[2]) + (x1 * x1 * wgt[1]) +
                  (x0 * x0 * wgt[0]);
    }

    float x;
    switch (last - last_aligned) {
    case 3:
        x = lhs[2] - rhs[2];
        result += (x * x * wgt[2]);
        /* FALLTHRU */
    case 2:
        x = lhs[1] - rhs[1];
        result += (x * x * wgt[1]);
        /* FALLTHRU */
    case 1:
        x = lhs[0] - rhs[0];
        result += (x * x * wgt[0]);
    }
    return result;
}

template <typename T>
static inline float manhattan_distance(const T *lhs, const T *rhs, size_t size)
{
    const T *last = lhs + size;
    const T *last_aligned = lhs + ((size >> 2) << 2);
    float result = 0.0f;

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        result += (fast_abs(lhs[3] - rhs[3]) + fast_abs(lhs[2] - rhs[2]) +
                   fast_abs(lhs[1] - rhs[1]) + fast_abs(lhs[0] - rhs[0]));
    }
    switch (last - last_aligned) {
    case 3:
        result += fast_abs(lhs[2] - rhs[2]);
        /* FALLTHRU */
    case 2:
        result += fast_abs(lhs[1] - rhs[1]);
        /* FALLTHRU */
    case 1:
        result += fast_abs(lhs[0] - rhs[0]);
    }
    return result;
}

template <typename T>
static inline float chebyshev_distance(const T *lhs, const T *rhs, size_t size)
{
    const T *last = lhs + size;
    const T *last_aligned = lhs + ((size >> 2) << 2);
    float result = 0.0f;

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        float x3 = fast_abs(lhs[3] - rhs[3]);
        float x2 = fast_abs(lhs[2] - rhs[2]);
        float x1 = fast_abs(lhs[1] - rhs[1]);
        float x0 = fast_abs(lhs[0] - rhs[0]);

        if (x1 < x3) {
            x1 = x3;
        }
        if (x0 < x2) {
            x0 = x2;
        }
        if (x0 < x1) {
            x0 = x1;
        }
        if (result < x0) {
            result = x0;
        }
    }

    float x;
    switch (last - last_aligned) {
    case 3:
        x = fast_abs(lhs[2] - rhs[2]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 2:
        x = fast_abs(lhs[1] - rhs[1]);
        if (result < x) {
            result = x;
        }
        /* FALLTHRU */
    case 1:
        x = fast_abs(lhs[0] - rhs[0]);
        if (result < x) {
            result = x;
        }
    }
    return result;
}

template <typename T>
static inline float cosine_distance(const T *lhs, const T *rhs, size_t size)
{
    const T *last = lhs + size;
    const T *last_aligned = lhs + ((size >> 2) << 2);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        T x31 = lhs[3];
        T x32 = rhs[3];
        T x21 = lhs[2];
        T x22 = rhs[2];
        T x11 = lhs[1];
        T x12 = rhs[1];
        T x01 = lhs[0];
        T x02 = rhs[0];
        sum1 += (float)(x31 * x32) + (float)(x21 * x22) + (float)(x11 * x12) +
                (float)(x01 * x02);
        sum2 += (float)(x31 * x31) + (float)(x21 * x21) + (float)(x11 * x11) +
                (float)(x01 * x01);
        sum3 += (float)(x32 * x32) + (float)(x22 * x22) + (float)(x12 * x12) +
                (float)(x02 * x02);
    }

    T x1, x2;
    switch (last - last_aligned) {
    case 3:
        x1 = lhs[2];
        x2 = rhs[2];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1];
        x2 = rhs[1];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0];
        x2 = rhs[0];
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
    }
    return (1.0f - sum1 / (fast_sqrt(sum2) * fast_sqrt(sum3)));
}

template <typename T>
static inline float canberra_distance(const T *lhs, const T *rhs, size_t size)
{
    const T *last = lhs + size;
    const T *last_aligned = lhs + ((size >> 2) << 2);
    float result = 0.0f;

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        T x31 = lhs[3];
        T x32 = rhs[3];
        T x21 = lhs[2];
        T x22 = rhs[2];
        T x11 = lhs[1];
        T x12 = rhs[1];
        T x01 = lhs[0];
        T x02 = rhs[0];
        result +=
            (float)fast_abs(x31 - x32) / (float)(fast_abs(x31) + fast_abs(x32));
        result +=
            (float)fast_abs(x21 - x22) / (float)(fast_abs(x21) + fast_abs(x22));
        result +=
            (float)fast_abs(x11 - x12) / (float)(fast_abs(x11) + fast_abs(x12));
        result +=
            (float)fast_abs(x01 - x02) / (float)(fast_abs(x01) + fast_abs(x02));
    }

    T x1, x2;
    switch (last - last_aligned) {
    case 3:
        x1 = lhs[2];
        x2 = rhs[2];
        result +=
            (float)fast_abs(x1 - x2) / (float)(fast_abs(x1) + fast_abs(x2));
        /* FALLTHRU */
    case 2:
        x1 = lhs[1];
        x2 = rhs[1];
        result +=
            (float)fast_abs(x1 - x2) / (float)(fast_abs(x1) + fast_abs(x2));
        /* FALLTHRU */
    case 1:
        x1 = lhs[0];
        x2 = rhs[0];
        result +=
            (float)fast_abs(x1 - x2) / (float)(fast_abs(x1) + fast_abs(x2));
    }
    return result;
}

template <typename T>
static inline float bray_curtis_distance(const T *lhs, const T *rhs,
                                         size_t size)
{
    const T *last = lhs + size;
    const T *last_aligned = lhs + ((size >> 2) << 2);
    float sum1 = 0.0f;
    float sum2 = 0.0f;

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        T x31 = lhs[3];
        T x32 = rhs[3];
        T x21 = lhs[2];
        T x22 = rhs[2];
        T x11 = lhs[1];
        T x12 = rhs[1];
        T x01 = lhs[0];
        T x02 = rhs[0];
        sum1 += (fast_abs(x31 - x32) + fast_abs(x21 - x22) +
                 fast_abs(x11 - x12) + fast_abs(x01 - x02));
        sum2 += (fast_abs(x31 + x32) + fast_abs(x21 + x22) +
                 fast_abs(x11 + x12) + fast_abs(x01 + x02));
    }

    T x1, x2;
    switch (last - last_aligned) {
    case 3:
        x1 = lhs[2];
        x2 = rhs[2];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1];
        x2 = rhs[1];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0];
        x2 = rhs[0];
        sum1 += fast_abs(x1 - x2);
        sum2 += fast_abs(x1 + x2);
    }
    return (sum1 / sum2);
}

template <typename T>
static inline float correlation_distance(const T *lhs, const T *rhs,
                                         size_t size)
{
    const T *last = lhs + size;
    const T *last_aligned = lhs + ((size >> 2) << 2);
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float avg1 = horizontal_mean(lhs, size);
    float avg2 = horizontal_mean(rhs, size);

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        float x31 = lhs[3] - avg1;
        float x32 = rhs[3] - avg2;
        float x21 = lhs[2] - avg1;
        float x22 = rhs[2] - avg2;
        float x11 = lhs[1] - avg1;
        float x12 = rhs[1] - avg2;
        float x01 = lhs[0] - avg1;
        float x02 = rhs[0] - avg2;
        sum1 += (x31 * x32 + x21 * x22 + x11 * x12 + x01 * x02);
        sum2 += (x31 * x31 + x21 * x21 + x11 * x11 + x01 * x01);
        sum3 += (x32 * x32 + x22 * x22 + x12 * x12 + x02 * x02);
    }

    float x1, x2;
    switch (last - last_aligned) {
    case 3:
        x1 = lhs[2] - avg1;
        x2 = rhs[2] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 2:
        x1 = lhs[1] - avg1;
        x2 = rhs[1] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
        /* FALLTHRU */
    case 1:
        x1 = lhs[0] - avg1;
        x2 = rhs[0] - avg2;
        sum1 += (x1 * x2);
        sum2 += (x1 * x1);
        sum3 += (x2 * x2);
    }
    return (1.0f - sum1 / (fast_sqrt(sum2) * fast_sqrt(sum3)));
}

template <typename T>
static inline float binary_distance(const T *lhs, const T *rhs, size_t size)
{
    return (memcmp(lhs, rhs, sizeof(T) * size) != 0 ? 1.0f : 0.0f);
}

static inline float binary_distance(const float *lhs, const float *rhs,
                                    size_t size)
{
    const float *last = lhs + size;
    const float *last_aligned = lhs + ((size >> 2) << 2);

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        if (fast_abs(lhs[3] - rhs[3]) > FLT_EPSILON) {
            return 1.0f;
        }
        if (fast_abs(lhs[2] - rhs[2]) > FLT_EPSILON) {
            return 1.0f;
        }
        if (fast_abs(lhs[1] - rhs[1]) > FLT_EPSILON) {
            return 1.0f;
        }
        if (fast_abs(lhs[0] - rhs[0]) > FLT_EPSILON) {
            return 1.0f;
        }
    }
    switch (last - last_aligned) {
    case 3:
        if (fast_abs(lhs[2] - rhs[2]) > FLT_EPSILON) {
            return 1.0f;
        }
        /* FALLTHRU */
    case 2:
        if (fast_abs(lhs[1] - rhs[1]) > FLT_EPSILON) {
            return 1.0f;
        }
        /* FALLTHRU */
    case 1:
        if (fast_abs(lhs[0] - rhs[0]) > FLT_EPSILON) {
            return 1.0f;
        }
    }
    return 0.0f;
}

template <typename T>
static inline float inner_product(const T *lhs, const T *rhs, size_t size)
{
    const T *last = lhs + size;
    const T *last_aligned = lhs + ((size >> 2) << 2);
    float sum = 0.0f;

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        sum += (float)(lhs[3] * rhs[3]) + (float)(lhs[2] * rhs[2]) +
               (float)(lhs[1] * rhs[1]) + (float)(lhs[0] * rhs[0]);
    }

    switch (last - last_aligned) {
    case 3:
        sum += (lhs[2] * rhs[2]);
        /* FALLTHRU */
    case 2:
        sum += (lhs[1] * rhs[1]);
        /* FALLTHRU */
    case 1:
        sum += (lhs[0] * rhs[0]);
    }
    return sum;
}

namespace mercury {
namespace internal {

float SquaredEuclideanDistance(const float *lhs, const float *rhs, size_t size)
{
#if defined(__AVX512F__)
    if (size > 15 && CpuFeatures::AVX512F()) {
        return squared_euclidean_distance_v512(lhs, rhs, size);
    }
#endif
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return squared_euclidean_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return squared_euclidean_distance_v128(lhs, rhs, size);
    }
#endif
    return squared_euclidean_distance(lhs, rhs, size);
}

float EuclideanDistance(const float *lhs, const float *rhs, size_t size)
{
#if defined(__AVX512F__)
    if (size > 15 && CpuFeatures::AVX512F()) {
        return fast_sqrt(squared_euclidean_distance_v512(lhs, rhs, size));
    }
#endif
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return fast_sqrt(squared_euclidean_distance_v256(lhs, rhs, size));
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return fast_sqrt(squared_euclidean_distance_v128(lhs, rhs, size));
    }
#endif
    return fast_sqrt(squared_euclidean_distance(lhs, rhs, size));
}

float NormalizedEuclideanDistance(const float *lhs, const float *rhs,
                                  size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return fast_sqrt(
            normalized_squared_euclidean_distance_v256(lhs, rhs, size));
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return fast_sqrt(
            normalized_squared_euclidean_distance_v128(lhs, rhs, size));
    }
#endif
    return fast_sqrt(normalized_squared_euclidean_distance(lhs, rhs, size));
}

float NormalizedSquaredEuclideanDistance(const float *lhs, const float *rhs,
                                         size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return normalized_squared_euclidean_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return normalized_squared_euclidean_distance_v128(lhs, rhs, size);
    }
#endif
    return normalized_squared_euclidean_distance(lhs, rhs, size);
}

float SquaredEuclideanDistance(const float *lhs, const float *rhs,
                               const float *wgt, size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return weighted_squared_euclidean_distance_v256(lhs, rhs, wgt, size);
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return weighted_squared_euclidean_distance_v128(lhs, rhs, wgt, size);
    }
#endif
    return weighted_squared_euclidean_distance(lhs, rhs, wgt, size);
}

float EuclideanDistance(const float *lhs, const float *rhs, const float *wgt,
                        size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return fast_sqrt(
            weighted_squared_euclidean_distance_v256(lhs, rhs, wgt, size));
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return fast_sqrt(
            weighted_squared_euclidean_distance_v128(lhs, rhs, wgt, size));
    }
#endif
    return fast_sqrt(weighted_squared_euclidean_distance(lhs, rhs, wgt, size));
}

float ManhattanDistance(const float *lhs, const float *rhs, size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return manhattan_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return manhattan_distance_v128(lhs, rhs, size);
    }
#endif
    return manhattan_distance(lhs, rhs, size);
}

float ChebyshevDistance(const float *lhs, const float *rhs, size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return chebyshev_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return chebyshev_distance_v128(lhs, rhs, size);
    }
#endif
    return chebyshev_distance(lhs, rhs, size);
}

float CosineDistance(const float *lhs, const float *rhs, size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return cosine_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return cosine_distance_v128(lhs, rhs, size);
    }
#endif
    return cosine_distance(lhs, rhs, size);
}

float CanberraDistance(const float *lhs, const float *rhs, size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return canberra_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return canberra_distance_v128(lhs, rhs, size);
    }
#endif
    return canberra_distance(lhs, rhs, size);
}

float BrayCurtisDistance(const float *lhs, const float *rhs, size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return bray_curtis_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return bray_curtis_distance_v128(lhs, rhs, size);
    }
#endif
    return bray_curtis_distance(lhs, rhs, size);
}

float CorrelationDistance(const float *lhs, const float *rhs, size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return correlation_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return correlation_distance_v128(lhs, rhs, size);
    }
#endif
    return correlation_distance(lhs, rhs, size);
}

float BinaryDistance(const float *lhs, const float *rhs, size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return binary_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return binary_distance_v128(lhs, rhs, size);
    }
#endif
    return binary_distance(lhs, rhs, size);
}

float InnerProduct(const float *lhs, const float *rhs, size_t size)
{
#if defined(__AVX__)
    if (size > 7 && CpuFeatures::AVX()) {
        return inner_product_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE__)
    if (size > 3 && CpuFeatures::SSE()) {
        return inner_product_v128(lhs, rhs, size);
    }
#endif
    return inner_product(lhs, rhs, size);
}

float SquaredEuclideanDistance(const int16_t *lhs, const int16_t *rhs,
                               size_t size)
{
#if defined(__AVX2__)
    if (size > 15 && CpuFeatures::AVX2()) {
        return squared_euclidean_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE3__)
    if (size > 7 && CpuFeatures::SSE3()) {
        return squared_euclidean_distance_v128(lhs, rhs, size);
    }
#elif defined(__SSE2__)
    if (size > 7 && CpuFeatures::SSE2()) {
        return squared_euclidean_distance_v128(lhs, rhs, size);
    }
#endif
    return squared_euclidean_distance(lhs, rhs, size);
}

float EuclideanDistance(const int16_t *lhs, const int16_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (size > 15 && CpuFeatures::AVX2()) {
        return fast_sqrt(squared_euclidean_distance_v256(lhs, rhs, size));
    }
#endif
#if defined(__SSE3__)
    if (size > 7 && CpuFeatures::SSE3()) {
        return fast_sqrt(squared_euclidean_distance_v128(lhs, rhs, size));
    }
#elif defined(__SSE2__)
    if (size > 7 && CpuFeatures::SSE2()) {
        return fast_sqrt(squared_euclidean_distance_v128(lhs, rhs, size));
    }
#endif
    return fast_sqrt(squared_euclidean_distance(lhs, rhs, size));
}

float NormalizedEuclideanDistance(const int16_t *lhs, const int16_t *rhs,
                                  size_t size)
{
    return fast_sqrt(normalized_squared_euclidean_distance(lhs, rhs, size));
}

float NormalizedSquaredEuclideanDistance(const int16_t *lhs, const int16_t *rhs,
                                         size_t size)
{
    return normalized_squared_euclidean_distance(lhs, rhs, size);
}

float SquaredEuclideanDistance(const int16_t *lhs, const int16_t *rhs,
                               const float *wgt, size_t size)
{
#if defined(__AVX2__)
    if (size > 15 && CpuFeatures::AVX2()) {
        return weighted_squared_euclidean_distance_v256(lhs, rhs, wgt, size);
    }
#endif
#if defined(__SSE3__)
    if (size > 7 && CpuFeatures::SSE3()) {
        return weighted_squared_euclidean_distance_v128(lhs, rhs, wgt, size);
    }
#elif defined(__SSE2__)
    if (size > 7 && CpuFeatures::SSE2()) {
        return weighted_squared_euclidean_distance_v128(lhs, rhs, wgt, size);
    }
#endif
    return weighted_squared_euclidean_distance(lhs, rhs, wgt, size);
}

float EuclideanDistance(const int16_t *lhs, const int16_t *rhs,
                        const float *wgt, size_t size)
{
#if defined(__AVX2__)
    if (size > 15 && CpuFeatures::AVX2()) {
        return fast_sqrt(
            weighted_squared_euclidean_distance_v256(lhs, rhs, wgt, size));
    }
#endif
#if defined(__SSE3__)
    if (size > 7 && CpuFeatures::SSE3()) {
        return fast_sqrt(
            weighted_squared_euclidean_distance_v128(lhs, rhs, wgt, size));
    }
#elif defined(__SSE2__)
    if (size > 7 && CpuFeatures::SSE2()) {
        return fast_sqrt(
            weighted_squared_euclidean_distance_v128(lhs, rhs, wgt, size));
    }
#endif
    return fast_sqrt(weighted_squared_euclidean_distance(lhs, rhs, wgt, size));
}

float ManhattanDistance(const int16_t *lhs, const int16_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (size > 15 && CpuFeatures::AVX2()) {
        return manhattan_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE3__)
    if (size > 7 && CpuFeatures::SSE3()) {
        return manhattan_distance_v128(lhs, rhs, size);
    }
#elif defined(__SSE2__)
    if (size > 7 && CpuFeatures::SSE2()) {
        return manhattan_distance_v128(lhs, rhs, size);
    }
#endif
    return manhattan_distance(lhs, rhs, size);
}

float ChebyshevDistance(const int16_t *lhs, const int16_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (size > 15 && CpuFeatures::AVX2()) {
        return chebyshev_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE4_1__)
    if (size > 7 && CpuFeatures::SSE4_1()) {
        return chebyshev_distance_v128(lhs, rhs, size);
    }
#endif
    return chebyshev_distance(lhs, rhs, size);
}

float CosineDistance(const int16_t *lhs, const int16_t *rhs, size_t size)
{
#if defined(__SSE3__)
    if (size > 7 && CpuFeatures::SSE3()) {
        return cosine_distance_v128(lhs, rhs, size);
    }
#elif defined(__SSE2__)
    if (size > 7 && CpuFeatures::SSE2()) {
        return cosine_distance_v128(lhs, rhs, size);
    }
#endif
    return cosine_distance(lhs, rhs, size);
}

float CanberraDistance(const int16_t *lhs, const int16_t *rhs, size_t size)
{
    return canberra_distance(lhs, rhs, size);
}

float BrayCurtisDistance(const int16_t *lhs, const int16_t *rhs, size_t size)
{
    return bray_curtis_distance(lhs, rhs, size);
}

float CorrelationDistance(const int16_t *lhs, const int16_t *rhs, size_t size)
{
    return correlation_distance(lhs, rhs, size);
}

float BinaryDistance(const int16_t *lhs, const int16_t *rhs, size_t size)
{
    return binary_distance(lhs, rhs, size);
}

float InnerProduct(const int16_t *lhs, const int16_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (size > 15 && CpuFeatures::AVX2()) {
        return inner_product_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE3__)
    if (size > 7 && CpuFeatures::SSE3()) {
        return inner_product_v128(lhs, rhs, size);
    }
#elif defined(__SSE2__)
    if (size > 7 && CpuFeatures::SSE2()) {
        return inner_product_v128(lhs, rhs, size);
    }
#endif
    return inner_product(lhs, rhs, size);
}

float SquaredEuclideanDistance(const int8_t *lhs, const int8_t *rhs,
                               size_t size)
{
#if defined(__AVX2__)
    if (size > 31 && CpuFeatures::AVX2()) {
        return squared_euclidean_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE4_1__)
    if (size > 15 && CpuFeatures::SSE4_1()) {
        return squared_euclidean_distance_v128(lhs, rhs, size);
    }
#endif
    return squared_euclidean_distance(lhs, rhs, size);
}

float EuclideanDistance(const int8_t *lhs, const int8_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (size > 31 && CpuFeatures::AVX2()) {
        return fast_sqrt(squared_euclidean_distance_v256(lhs, rhs, size));
    }
#endif
#if defined(__SSE4_1__)
    if (size > 15 && CpuFeatures::SSE4_1()) {
        return fast_sqrt(squared_euclidean_distance_v128(lhs, rhs, size));
    }
#endif
    return fast_sqrt(squared_euclidean_distance(lhs, rhs, size));
}

float NormalizedEuclideanDistance(const int8_t *lhs, const int8_t *rhs,
                                  size_t size)
{
    return fast_sqrt(normalized_squared_euclidean_distance(lhs, rhs, size));
}

float NormalizedSquaredEuclideanDistance(const int8_t *lhs, const int8_t *rhs,
                                         size_t size)
{
    return normalized_squared_euclidean_distance(lhs, rhs, size);
}

float SquaredEuclideanDistance(const int8_t *lhs, const int8_t *rhs,
                               const float *wgt, size_t size)
{
    return weighted_squared_euclidean_distance(lhs, rhs, wgt, size);
}

float EuclideanDistance(const int8_t *lhs, const int8_t *rhs, const float *wgt,
                        size_t size)
{
    return fast_sqrt(weighted_squared_euclidean_distance(lhs, rhs, wgt, size));
}

float ManhattanDistance(const int8_t *lhs, const int8_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (size > 31 && CpuFeatures::AVX2()) {
        return manhattan_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE4_1__)
    if (size > 15 && CpuFeatures::SSE4_1()) {
        return manhattan_distance_v128(lhs, rhs, size);
    }
#endif
    return manhattan_distance(lhs, rhs, size);
}

float ChebyshevDistance(const int8_t *lhs, const int8_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (size > 31 && CpuFeatures::AVX2()) {
        return chebyshev_distance_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE4_1__)
    if (size > 15 && CpuFeatures::SSE4_1()) {
        return chebyshev_distance_v128(lhs, rhs, size);
    }
#endif
    return chebyshev_distance(lhs, rhs, size);
}

float CosineDistance(const int8_t *lhs, const int8_t *rhs, size_t size)
{
#if defined(__SSE4_1__)
    if (size > 15 && CpuFeatures::SSE4_1()) {
        return cosine_distance_v128(lhs, rhs, size);
    }
#endif
    return cosine_distance(lhs, rhs, size);
}

float CanberraDistance(const int8_t *lhs, const int8_t *rhs, size_t size)
{
    return canberra_distance(lhs, rhs, size);
}

float BrayCurtisDistance(const int8_t *lhs, const int8_t *rhs, size_t size)
{
    return bray_curtis_distance(lhs, rhs, size);
}

float CorrelationDistance(const int8_t *lhs, const int8_t *rhs, size_t size)
{
    return correlation_distance(lhs, rhs, size);
}

float BinaryDistance(const int8_t *lhs, const int8_t *rhs, size_t size)
{
    return binary_distance(lhs, rhs, size);
}

float InnerProduct(const int8_t *lhs, const int8_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (size > 31 && CpuFeatures::AVX2()) {
        return inner_product_v256(lhs, rhs, size);
    }
#endif
#if defined(__SSE4_1__)
    if (size > 15 && CpuFeatures::SSE4_1()) {
        return inner_product_v128(lhs, rhs, size);
    }
#endif
    return inner_product(lhs, rhs, size);
}

} // namespace internal
} // namespace mercury
