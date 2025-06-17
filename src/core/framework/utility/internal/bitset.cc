/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     bitset.cc
 *   \author   Hechong.xyf
 *   \date     Dec 2017
 *   \version  1.0.0
 *   \brief    Implementation of Bitset Utility
 */

#include "bitset.h"
#include "cpu_features.h"

// #undef __AVX2__
// #undef __SSE2__
// #undef __SSE3__
// #undef __SSE4_1__
// #undef __SSE4_2__

#ifndef __SSE4_2__
#define bitset_popcount32 platform_popcount32
#define bitset_popcount64 platform_popcount64
#else
#define bitset_popcount32 _mm_popcnt_u32
#define bitset_popcount64 _mm_popcnt_u64
#endif // !__SSE4_2__

#if defined(__AVX2__)
static inline void bitset_and_v256(uint32_t *lhs, const uint32_t *rhs,
                                   size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
            __m256i ymm1 = _mm256_load_si256((__m256i *)rhs);
            _mm256_store_si256((__m256i *)lhs, _mm256_and_si256(ymm1, ymm0));
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
            __m256i ymm1 = _mm256_loadu_si256((__m256i *)rhs);
            _mm256_storeu_si256((__m256i *)lhs, _mm256_and_si256(ymm1, ymm0));
        }
    }
    switch (last - last_aligned) {
    case 7:
        lhs[6] &= rhs[6];
        /* FALLTHRU */
    case 6:
        lhs[5] &= rhs[5];
        /* FALLTHRU */
    case 5:
        lhs[4] &= rhs[4];
        /* FALLTHRU */
    case 4:
        lhs[3] &= rhs[3];
        /* FALLTHRU */
    case 3:
        lhs[2] &= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] &= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] &= rhs[0];
    }
}

static inline void bitset_andnot_v256(uint32_t *lhs, const uint32_t *rhs,
                                      size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
            __m256i ymm1 = _mm256_load_si256((__m256i *)rhs);
            _mm256_store_si256((__m256i *)lhs, _mm256_andnot_si256(ymm1, ymm0));
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
            __m256i ymm1 = _mm256_loadu_si256((__m256i *)rhs);
            _mm256_storeu_si256((__m256i *)lhs,
                                _mm256_andnot_si256(ymm1, ymm0));
        }
    }
    switch (last - last_aligned) {
    case 7:
        lhs[6] &= ~rhs[6];
        /* FALLTHRU */
    case 6:
        lhs[5] &= ~rhs[5];
        /* FALLTHRU */
    case 5:
        lhs[4] &= ~rhs[4];
        /* FALLTHRU */
    case 4:
        lhs[3] &= ~rhs[3];
        /* FALLTHRU */
    case 3:
        lhs[2] &= ~rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] &= ~rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] &= ~rhs[0];
    }
}

static inline void bitset_or_v256(uint32_t *lhs, const uint32_t *rhs,
                                  size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
            __m256i ymm1 = _mm256_load_si256((__m256i *)rhs);
            _mm256_store_si256((__m256i *)lhs, _mm256_or_si256(ymm1, ymm0));
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
            __m256i ymm1 = _mm256_loadu_si256((__m256i *)rhs);
            _mm256_storeu_si256((__m256i *)lhs, _mm256_or_si256(ymm1, ymm0));
        }
    }
    switch (last - last_aligned) {
    case 7:
        lhs[6] |= rhs[6];
        /* FALLTHRU */
    case 6:
        lhs[5] |= rhs[5];
        /* FALLTHRU */
    case 5:
        lhs[4] |= rhs[4];
        /* FALLTHRU */
    case 4:
        lhs[3] |= rhs[3];
        /* FALLTHRU */
    case 3:
        lhs[2] |= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] |= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] |= rhs[0];
    }
}

static inline void bitset_xor_v256(uint32_t *lhs, const uint32_t *rhs,
                                   size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
            __m256i ymm1 = _mm256_load_si256((__m256i *)rhs);
            _mm256_store_si256((__m256i *)lhs, _mm256_xor_si256(ymm1, ymm0));
        }
    } else {
        for (; lhs != last_aligned; lhs += 8, rhs += 8) {
            __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
            __m256i ymm1 = _mm256_loadu_si256((__m256i *)rhs);
            _mm256_storeu_si256((__m256i *)lhs, _mm256_xor_si256(ymm1, ymm0));
        }
    }
    switch (last - last_aligned) {
    case 7:
        lhs[6] ^= rhs[6];
        /* FALLTHRU */
    case 6:
        lhs[5] ^= rhs[5];
        /* FALLTHRU */
    case 5:
        lhs[4] ^= rhs[4];
        /* FALLTHRU */
    case 4:
        lhs[3] ^= rhs[3];
        /* FALLTHRU */
    case 3:
        lhs[2] ^= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] ^= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] ^= rhs[0];
    }
}

static inline void bitset_not_v256(uint32_t *lhs, size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 3) << 3);
    static const __m256i mask = _mm256_set1_epi32(0xffffffffu);

    if (((uintptr_t)lhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 8) {
            _mm256_store_si256(
                (__m256i *)lhs,
                _mm256_andnot_si256(_mm256_load_si256((__m256i *)lhs), mask));
        }
    } else {
        for (; lhs != last_aligned; lhs += 8) {
            _mm256_storeu_si256(
                (__m256i *)lhs,
                _mm256_andnot_si256(_mm256_loadu_si256((__m256i *)lhs), mask));
        }
    }
    switch (last - last_aligned) {
    case 7:
        lhs[6] = ~lhs[6];
        /* FALLTHRU */
    case 6:
        lhs[5] = ~lhs[5];
        /* FALLTHRU */
    case 5:
        lhs[4] = ~lhs[4];
        /* FALLTHRU */
    case 4:
        lhs[3] = ~lhs[3];
        /* FALLTHRU */
    case 3:
        lhs[2] = ~lhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] = ~lhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] = ~lhs[0];
    }
}

bool bitset_test_all_v256(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
    static const __m256i mask = _mm256_set1_epi32(0xffffffffu);

    if (((uintptr_t)lhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 8) {
            __m256i neq =
                _mm256_xor_si256(_mm256_load_si256((__m256i *)lhs), mask);
            if (!_mm256_testz_si256(neq, neq)) {
                return false;
            }
        }
    } else {
        for (; lhs != last_aligned; lhs += 8) {
            __m256i neq =
                _mm256_xor_si256(_mm256_loadu_si256((__m256i *)lhs), mask);
            if (!_mm256_testz_si256(neq, neq)) {
                return false;
            }
        }
    }
    switch (last - last_aligned) {
    case 7:
        if (lhs[6] != 0xffffffffu) {
            return false;
        }
        /* FALLTHRU */
    case 6:
        if (lhs[5] != 0xffffffffu) {
            return false;
        }
        /* FALLTHRU */
    case 5:
        if (lhs[4] != 0xffffffffu) {
            return false;
        }
        /* FALLTHRU */
    case 4:
        if (lhs[3] != 0xffffffffu) {
            return false;
        }
        /* FALLTHRU */
    case 3:
        if (lhs[2] != 0xffffffffu) {
            return false;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != 0xffffffffu) {
            return false;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != 0xffffffffu) {
            return false;
        }
    }
    return true;
}

bool bitset_test_any_v256(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    if (((uintptr_t)lhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 8) {
            __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
            if (!_mm256_testz_si256(ymm0, ymm0)) {
                return true;
            }
        }
    } else {
        for (; lhs != last_aligned; lhs += 8) {
            __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
            if (!_mm256_testz_si256(ymm0, ymm0)) {
                return true;
            }
        }
    }
    switch (last - last_aligned) {
    case 7:
        if (lhs[6] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 6:
        if (lhs[5] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 5:
        if (lhs[4] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 4:
        if (lhs[3] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 3:
        if (lhs[2] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != 0u) {
            return true;
        }
    }
    return false;
}

bool bitset_test_none_v256(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    if (((uintptr_t)lhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 8) {
            __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
            if (!_mm256_testz_si256(ymm0, ymm0)) {
                return false;
            }
        }
    } else {
        for (; lhs != last_aligned; lhs += 8) {
            __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
            if (!_mm256_testz_si256(ymm0, ymm0)) {
                return false;
            }
        }
    }
    switch (last - last_aligned) {
    case 7:
        if (lhs[6] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 6:
        if (lhs[5] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 5:
        if (lhs[4] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 4:
        if (lhs[3] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 3:
        if (lhs[2] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != 0u) {
            return false;
        }
    }
    return true;
}

static inline size_t horizontal_add_v256(__m256i v)
{
    __m256i x1 = _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
    __m256i x2 = _mm256_add_epi64(v, x1);
    __m128i x3 = _mm256_extractf128_si256(x2, 1);
    __m128i x4 = _mm_add_epi64(_mm256_extractf128_si256(x2, 0), x3);
    return _mm_cvtsi128_si32(x4);
}

static inline __m256i bitset_popcount_v256(__m256i v)
{
    static const __m256i lookup =
        _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1,
                         1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    static const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    __m256i total = _mm256_add_epi8(popcnt1, popcnt2);
    return _mm256_sad_epu8(total, _mm256_setzero_si256());
}

static inline size_t bitset_cardinality_v256(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 4) << 4);
    size_t count = 0;

    __m256i ymm_sum1 = _mm256_setzero_si256();
    __m256i ymm_sum2 = _mm256_setzero_si256();

    if (((uintptr_t)lhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16) {
            __m256i ymm0 = _mm256_load_si256((__m256i *)(lhs));
            __m256i ymm1 = _mm256_load_si256((__m256i *)(lhs + 8));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            ymm_sum2 = _mm256_add_epi64(ymm_sum2, bitset_popcount_v256(ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256i ymm0 = _mm256_load_si256((__m256i *)(lhs));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            lhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16) {
            __m256i ymm0 = _mm256_loadu_si256((__m256i *)(lhs));
            __m256i ymm1 = _mm256_loadu_si256((__m256i *)(lhs + 8));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            ymm_sum2 = _mm256_add_epi64(ymm_sum2, bitset_popcount_v256(ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256i ymm0 = _mm256_loadu_si256((__m256i *)(lhs));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            lhs += 8;
        }
    }
    count = horizontal_add_v256(_mm256_add_epi64(ymm_sum1, ymm_sum2));

    switch (last - lhs) {
    case 7:
        count += bitset_popcount32(lhs[6]);
        /* FALLTHRU */
    case 6:
        count += bitset_popcount32(lhs[5]);
        /* FALLTHRU */
    case 5:
        count += bitset_popcount32(lhs[4]);
        /* FALLTHRU */
    case 4:
        count += bitset_popcount32(lhs[3]);
        /* FALLTHRU */
    case 3:
        count += bitset_popcount32(lhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0]);
    }
    return count;
}

static inline size_t bitset_xor_cardinality_v256(const uint32_t *lhs,
                                                 const uint32_t *rhs,
                                                 size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 4) << 4);
    size_t count = 0;

    __m256i ymm_sum1 = _mm256_setzero_si256();
    __m256i ymm_sum2 = _mm256_setzero_si256();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i ymm0 =
                _mm256_xor_si256(_mm256_load_si256((__m256i *)(lhs)),
                                 _mm256_load_si256((__m256i *)(rhs)));
            __m256i ymm1 =
                _mm256_xor_si256(_mm256_load_si256((__m256i *)(lhs + 8)),
                                 _mm256_load_si256((__m256i *)(rhs + 8)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            ymm_sum2 = _mm256_add_epi64(ymm_sum2, bitset_popcount_v256(ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256i ymm0 =
                _mm256_xor_si256(_mm256_load_si256((__m256i *)(lhs)),
                                 _mm256_load_si256((__m256i *)(rhs)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i ymm0 =
                _mm256_xor_si256(_mm256_loadu_si256((__m256i *)(lhs)),
                                 _mm256_loadu_si256((__m256i *)(rhs)));
            __m256i ymm1 =
                _mm256_xor_si256(_mm256_loadu_si256((__m256i *)(lhs + 8)),
                                 _mm256_loadu_si256((__m256i *)(rhs + 8)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            ymm_sum2 = _mm256_add_epi64(ymm_sum2, bitset_popcount_v256(ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256i ymm0 =
                _mm256_xor_si256(_mm256_loadu_si256((__m256i *)(lhs)),
                                 _mm256_loadu_si256((__m256i *)(rhs)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            lhs += 8;
            rhs += 8;
        }
    }
    count = horizontal_add_v256(_mm256_add_epi64(ymm_sum1, ymm_sum2));

    switch (last - lhs) {
    case 7:
        count += bitset_popcount32(lhs[6] ^ rhs[6]);
        /* FALLTHRU */
    case 6:
        count += bitset_popcount32(lhs[5] ^ rhs[5]);
        /* FALLTHRU */
    case 5:
        count += bitset_popcount32(lhs[4] ^ rhs[4]);
        /* FALLTHRU */
    case 4:
        count += bitset_popcount32(lhs[3] ^ rhs[3]);
        /* FALLTHRU */
    case 3:
        count += bitset_popcount32(lhs[2] ^ rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] ^ rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] ^ rhs[0]);
    }
    return count;
}

static inline size_t bitset_and_cardinality_v256(const uint32_t *lhs,
                                                 const uint32_t *rhs,
                                                 size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 4) << 4);
    size_t count = 0;

    __m256i ymm_sum1 = _mm256_setzero_si256();
    __m256i ymm_sum2 = _mm256_setzero_si256();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i ymm0 =
                _mm256_and_si256(_mm256_load_si256((__m256i *)(lhs)),
                                 _mm256_load_si256((__m256i *)(rhs)));
            __m256i ymm1 =
                _mm256_and_si256(_mm256_load_si256((__m256i *)(lhs + 8)),
                                 _mm256_load_si256((__m256i *)(rhs + 8)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            ymm_sum2 = _mm256_add_epi64(ymm_sum2, bitset_popcount_v256(ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256i ymm0 =
                _mm256_and_si256(_mm256_load_si256((__m256i *)(lhs)),
                                 _mm256_load_si256((__m256i *)(rhs)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i ymm0 =
                _mm256_and_si256(_mm256_loadu_si256((__m256i *)(lhs)),
                                 _mm256_loadu_si256((__m256i *)(rhs)));
            __m256i ymm1 =
                _mm256_and_si256(_mm256_loadu_si256((__m256i *)(lhs + 8)),
                                 _mm256_loadu_si256((__m256i *)(rhs + 8)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            ymm_sum2 = _mm256_add_epi64(ymm_sum2, bitset_popcount_v256(ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256i ymm0 =
                _mm256_and_si256(_mm256_loadu_si256((__m256i *)(lhs)),
                                 _mm256_loadu_si256((__m256i *)(rhs)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            lhs += 8;
            rhs += 8;
        }
    }
    count = horizontal_add_v256(_mm256_add_epi64(ymm_sum1, ymm_sum2));

    switch (last - lhs) {
    case 7:
        count += bitset_popcount32(lhs[6] & rhs[6]);
        /* FALLTHRU */
    case 6:
        count += bitset_popcount32(lhs[5] & rhs[5]);
        /* FALLTHRU */
    case 5:
        count += bitset_popcount32(lhs[4] & rhs[4]);
        /* FALLTHRU */
    case 4:
        count += bitset_popcount32(lhs[3] & rhs[3]);
        /* FALLTHRU */
    case 3:
        count += bitset_popcount32(lhs[2] & rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] & rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] & rhs[0]);
    }
    return count;
}

static inline size_t bitset_andnot_cardinality_v256(const uint32_t *lhs,
                                                    const uint32_t *rhs,
                                                    size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 4) << 4);
    size_t count = 0;

    __m256i ymm_sum1 = _mm256_setzero_si256();
    __m256i ymm_sum2 = _mm256_setzero_si256();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i ymm0 =
                _mm256_andnot_si256(_mm256_load_si256((__m256i *)(rhs)),
                                    _mm256_load_si256((__m256i *)(lhs)));
            __m256i ymm1 =
                _mm256_andnot_si256(_mm256_load_si256((__m256i *)(rhs + 8)),
                                    _mm256_load_si256((__m256i *)(lhs + 8)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            ymm_sum2 = _mm256_add_epi64(ymm_sum2, bitset_popcount_v256(ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256i ymm0 =
                _mm256_andnot_si256(_mm256_load_si256((__m256i *)(rhs)),
                                    _mm256_load_si256((__m256i *)(lhs)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i ymm0 =
                _mm256_andnot_si256(_mm256_loadu_si256((__m256i *)(rhs)),
                                    _mm256_loadu_si256((__m256i *)(lhs)));
            __m256i ymm1 =
                _mm256_andnot_si256(_mm256_loadu_si256((__m256i *)(rhs + 8)),
                                    _mm256_loadu_si256((__m256i *)(lhs + 8)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            ymm_sum2 = _mm256_add_epi64(ymm_sum2, bitset_popcount_v256(ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256i ymm0 =
                _mm256_andnot_si256(_mm256_loadu_si256((__m256i *)(rhs)),
                                    _mm256_loadu_si256((__m256i *)(lhs)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            lhs += 8;
            rhs += 8;
        }
    }
    count = horizontal_add_v256(_mm256_add_epi64(ymm_sum1, ymm_sum2));

    switch (last - lhs) {
    case 7:
        count += bitset_popcount32(lhs[6] & ~rhs[6]);
        /* FALLTHRU */
    case 6:
        count += bitset_popcount32(lhs[5] & ~rhs[5]);
        /* FALLTHRU */
    case 5:
        count += bitset_popcount32(lhs[4] & ~rhs[4]);
        /* FALLTHRU */
    case 4:
        count += bitset_popcount32(lhs[3] & ~rhs[3]);
        /* FALLTHRU */
    case 3:
        count += bitset_popcount32(lhs[2] & ~rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] & ~rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] & ~rhs[0]);
    }
    return count;
}

static inline size_t bitset_or_cardinality_v256(const uint32_t *lhs,
                                                const uint32_t *rhs,
                                                size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 4) << 4);
    size_t count = 0;

    __m256i ymm_sum1 = _mm256_setzero_si256();
    __m256i ymm_sum2 = _mm256_setzero_si256();

    if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i ymm0 = _mm256_or_si256(_mm256_load_si256((__m256i *)(lhs)),
                                           _mm256_load_si256((__m256i *)(rhs)));
            __m256i ymm1 =
                _mm256_or_si256(_mm256_load_si256((__m256i *)(lhs + 8)),
                                _mm256_load_si256((__m256i *)(rhs + 8)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            ymm_sum2 = _mm256_add_epi64(ymm_sum2, bitset_popcount_v256(ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256i ymm0 = _mm256_or_si256(_mm256_load_si256((__m256i *)(lhs)),
                                           _mm256_load_si256((__m256i *)(rhs)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            lhs += 8;
            rhs += 8;
        }
    } else {
        for (; lhs != last_aligned; lhs += 16, rhs += 16) {
            __m256i ymm0 =
                _mm256_or_si256(_mm256_loadu_si256((__m256i *)(lhs)),
                                _mm256_loadu_si256((__m256i *)(rhs)));
            __m256i ymm1 =
                _mm256_or_si256(_mm256_loadu_si256((__m256i *)(lhs + 8)),
                                _mm256_loadu_si256((__m256i *)(rhs + 8)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            ymm_sum2 = _mm256_add_epi64(ymm_sum2, bitset_popcount_v256(ymm1));
        }

        if ((last - last_aligned) > 7) {
            __m256i ymm0 =
                _mm256_or_si256(_mm256_loadu_si256((__m256i *)(lhs)),
                                _mm256_loadu_si256((__m256i *)(rhs)));
            ymm_sum1 = _mm256_add_epi64(ymm_sum1, bitset_popcount_v256(ymm0));
            lhs += 8;
            rhs += 8;
        }
    }
    count = horizontal_add_v256(_mm256_add_epi64(ymm_sum1, ymm_sum2));

    switch (last - lhs) {
    case 7:
        count += bitset_popcount32(lhs[6] | rhs[6]);
        /* FALLTHRU */
    case 6:
        count += bitset_popcount32(lhs[5] | rhs[5]);
        /* FALLTHRU */
    case 5:
        count += bitset_popcount32(lhs[4] | rhs[4]);
        /* FALLTHRU */
    case 4:
        count += bitset_popcount32(lhs[3] | rhs[3]);
        /* FALLTHRU */
    case 3:
        count += bitset_popcount32(lhs[2] | rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] | rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] | rhs[0]);
    }
    return count;
}
#endif // __AVX2__

#if defined(__SSE2__)
#ifndef __SSE3__
#define _mm_lddqu_si128 _mm_loadu_si128
#endif // !__SSE3__

static inline void bitset_and_v128(uint32_t *lhs, const uint32_t *rhs,
                                   size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 4, rhs += 4) {
            __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
            __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
            _mm_store_si128((__m128i *)lhs, _mm_and_si128(xmm1, xmm0));
        }
    } else {
        for (; lhs != last_aligned; lhs += 4, rhs += 4) {
            __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
            __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
            _mm_storeu_si128((__m128i *)lhs, _mm_and_si128(xmm1, xmm0));
        }
    }
    switch (last - last_aligned) {
    case 3:
        lhs[2] &= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] &= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] &= rhs[0];
    }
}

static inline void bitset_andnot_v128(uint32_t *lhs, const uint32_t *rhs,
                                      size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 4, rhs += 4) {
            __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
            __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
            _mm_store_si128((__m128i *)lhs, _mm_andnot_si128(xmm1, xmm0));
        }
    } else {
        for (; lhs != last_aligned; lhs += 4, rhs += 4) {
            __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
            __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
            _mm_storeu_si128((__m128i *)lhs, _mm_andnot_si128(xmm1, xmm0));
        }
    }
    switch (last - last_aligned) {
    case 3:
        lhs[2] &= ~rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] &= ~rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] &= ~rhs[0];
    }
}

static inline void bitset_or_v128(uint32_t *lhs, const uint32_t *rhs,
                                  size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 4, rhs += 4) {
            __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
            __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
            _mm_store_si128((__m128i *)lhs, _mm_or_si128(xmm1, xmm0));
        }
    } else {
        for (; lhs != last_aligned; lhs += 4, rhs += 4) {
            __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
            __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
            _mm_storeu_si128((__m128i *)lhs, _mm_or_si128(xmm1, xmm0));
        }
    }
    switch (last - last_aligned) {
    case 3:
        lhs[2] |= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] |= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] |= rhs[0];
    }
}

static inline void bitset_xor_v128(uint32_t *lhs, const uint32_t *rhs,
                                   size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 4, rhs += 4) {
            __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
            __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
            _mm_store_si128((__m128i *)lhs, _mm_xor_si128(xmm1, xmm0));
        }
    } else {
        for (; lhs != last_aligned; lhs += 4, rhs += 4) {
            __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
            __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
            _mm_storeu_si128((__m128i *)lhs, _mm_xor_si128(xmm1, xmm0));
        }
    }
    switch (last - last_aligned) {
    case 3:
        lhs[2] ^= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] ^= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] ^= rhs[0];
    }
}

static inline void bitset_not_v128(uint32_t *lhs, size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 2) << 2);
    static const __m128i mask = _mm_set1_epi32(0xffffffffu);

    if (((uintptr_t)lhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 4) {
            _mm_store_si128(
                (__m128i *)lhs,
                _mm_andnot_si128(_mm_load_si128((__m128i *)lhs), mask));
        }
    } else {
        for (; lhs != last_aligned; lhs += 4) {
            _mm_storeu_si128(
                (__m128i *)lhs,
                _mm_andnot_si128(_mm_lddqu_si128((__m128i *)lhs), mask));
        }
    }
    switch (last - last_aligned) {
    case 3:
        lhs[2] = ~lhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] = ~lhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] = ~lhs[0];
    }
}

bool bitset_test_all_v128(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
    static const __m128i mask = _mm_set1_epi32(0xffffffffu);

#ifndef __SSE4_1__
    if (((uintptr_t)lhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i eq = _mm_cmpeq_epi32(_mm_load_si128((__m128i *)lhs), mask);
            if (_mm_movemask_epi8(eq) != 0xffffu) {
                return false;
            }
        }
    } else {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i eq = _mm_cmpeq_epi32(_mm_lddqu_si128((__m128i *)lhs), mask);
            if (_mm_movemask_epi8(eq) != 0xffffu) {
                return false;
            }
        }
    }
#else
    if (((uintptr_t)lhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i neq = _mm_xor_si128(_mm_load_si128((__m128i *)lhs), mask);
            if (!_mm_testz_si128(neq, neq)) {
                return false;
            }
        }
    } else {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i neq = _mm_xor_si128(_mm_lddqu_si128((__m128i *)lhs), mask);
            if (!_mm_testz_si128(neq, neq)) {
                return false;
            }
        }
    }
#endif // !__SSE4_1__

    switch (last - last_aligned) {
    case 3:
        if (lhs[2] != 0xffffffffu) {
            return false;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != 0xffffffffu) {
            return false;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != 0xffffffffu) {
            return false;
        }
    }
    return true;
}

bool bitset_test_any_v128(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

#ifndef __SSE4_1__
    static const __m128i zero = _mm_setzero_si128();

    if (((uintptr_t)lhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i eq = _mm_cmpeq_epi32(_mm_load_si128((__m128i *)lhs), zero);
            if (_mm_movemask_epi8(eq) != 0xffffu) {
                return true;
            }
        }
    } else {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i eq = _mm_cmpeq_epi32(_mm_lddqu_si128((__m128i *)lhs), zero);
            if (_mm_movemask_epi8(eq) != 0xffffu) {
                return true;
            }
        }
    }
#else
    if (((uintptr_t)lhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
            if (!_mm_testz_si128(xmm0, xmm0)) {
                return true;
            }
        }
    } else {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
            if (!_mm_testz_si128(xmm0, xmm0)) {
                return true;
            }
        }
    }
#endif // !__SSE4_1__

    switch (last - last_aligned) {
    case 3:
        if (lhs[2] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != 0u) {
            return true;
        }
    }
    return false;
}

bool bitset_test_none_v128(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

#ifndef __SSE4_1__
    static __m128i zero = _mm_setzero_si128();

    if (((uintptr_t)lhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i eq = _mm_cmpeq_epi32(_mm_load_si128((__m128i *)lhs), zero);
            if (_mm_movemask_epi8(eq) != 0xffffu) {
                return false;
            }
        }
    } else {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i eq = _mm_cmpeq_epi32(_mm_lddqu_si128((__m128i *)lhs), zero);
            if (_mm_movemask_epi8(eq) != 0xffffu) {
                return false;
            }
        }
    }
#else
    if (((uintptr_t)lhs & 0xf) == 0) {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
            if (!_mm_testz_si128(xmm0, xmm0)) {
                return false;
            }
        }
    } else {
        for (; lhs != last_aligned; lhs += 4) {
            __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
            if (!_mm_testz_si128(xmm0, xmm0)) {
                return false;
            }
        }
    }
#endif // !__SSE4_1__

    switch (last - last_aligned) {
    case 3:
        if (lhs[2] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != 0u) {
            return false;
        }
    }
    return true;
}
#endif // __SSE2__

#if defined(PLATFORM_M64)
static inline void bitset_and(uint32_t *lhs, const uint32_t *rhs, size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
        *(uint64_t *)(&lhs[6]) &= *(uint64_t *)(&rhs[6]);
        *(uint64_t *)(&lhs[4]) &= *(uint64_t *)(&rhs[4]);
        *(uint64_t *)(&lhs[2]) &= *(uint64_t *)(&rhs[2]);
        *(uint64_t *)(&lhs[0]) &= *(uint64_t *)(&rhs[0]);
    }

    switch (last - last_aligned) {
    case 7:
        lhs[6] &= rhs[6];
        /* FALLTHRU */
    case 6:
        lhs[5] &= rhs[5];
        /* FALLTHRU */
    case 5:
        lhs[4] &= rhs[4];
        /* FALLTHRU */
    case 4:
        lhs[3] &= rhs[3];
        /* FALLTHRU */
    case 3:
        lhs[2] &= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] &= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] &= rhs[0];
    }
}

static inline void bitset_andnot(uint32_t *lhs, const uint32_t *rhs,
                                 size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
        *(uint64_t *)(&lhs[6]) &= ~(*(uint64_t *)(&rhs[6]));
        *(uint64_t *)(&lhs[4]) &= ~(*(uint64_t *)(&rhs[4]));
        *(uint64_t *)(&lhs[2]) &= ~(*(uint64_t *)(&rhs[2]));
        *(uint64_t *)(&lhs[0]) &= ~(*(uint64_t *)(&rhs[0]));
    }

    switch (last - last_aligned) {
    case 7:
        lhs[6] &= ~rhs[6];
        /* FALLTHRU */
    case 6:
        lhs[5] &= ~rhs[5];
        /* FALLTHRU */
    case 5:
        lhs[4] &= ~rhs[4];
        /* FALLTHRU */
    case 4:
        lhs[3] &= ~rhs[3];
        /* FALLTHRU */
    case 3:
        lhs[2] &= ~rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] &= ~rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] &= ~rhs[0];
    }
}

static inline void bitset_or(uint32_t *lhs, const uint32_t *rhs, size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
        *(uint64_t *)(&lhs[6]) |= *(uint64_t *)(&rhs[6]);
        *(uint64_t *)(&lhs[4]) |= *(uint64_t *)(&rhs[4]);
        *(uint64_t *)(&lhs[2]) |= *(uint64_t *)(&rhs[2]);
        *(uint64_t *)(&lhs[0]) |= *(uint64_t *)(&rhs[0]);
    }

    switch (last - last_aligned) {
    case 7:
        lhs[6] |= rhs[6];
        /* FALLTHRU */
    case 6:
        lhs[5] |= rhs[5];
        /* FALLTHRU */
    case 5:
        lhs[4] |= rhs[4];
        /* FALLTHRU */
    case 4:
        lhs[3] |= rhs[3];
        /* FALLTHRU */
    case 3:
        lhs[2] |= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] |= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] |= rhs[0];
    }
}

static inline void bitset_xor(uint32_t *lhs, const uint32_t *rhs, size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
        *(uint64_t *)(&lhs[6]) ^= *(uint64_t *)(&rhs[6]);
        *(uint64_t *)(&lhs[4]) ^= *(uint64_t *)(&rhs[4]);
        *(uint64_t *)(&lhs[2]) ^= *(uint64_t *)(&rhs[2]);
        *(uint64_t *)(&lhs[0]) ^= *(uint64_t *)(&rhs[0]);
    }

    switch (last - last_aligned) {
    case 7:
        lhs[6] ^= rhs[6];
        /* FALLTHRU */
    case 6:
        lhs[5] ^= rhs[5];
        /* FALLTHRU */
    case 5:
        lhs[4] ^= rhs[4];
        /* FALLTHRU */
    case 4:
        lhs[3] ^= rhs[3];
        /* FALLTHRU */
    case 3:
        lhs[2] ^= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] ^= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] ^= rhs[0];
    }
}

static inline void bitset_not(uint32_t *lhs, size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    for (; lhs != last_aligned; lhs += 8) {
        *(uint64_t *)(&lhs[6]) = ~(*(uint64_t *)(&lhs[6]));
        *(uint64_t *)(&lhs[4]) = ~(*(uint64_t *)(&lhs[4]));
        *(uint64_t *)(&lhs[2]) = ~(*(uint64_t *)(&lhs[2]));
        *(uint64_t *)(&lhs[0]) = ~(*(uint64_t *)(&lhs[0]));
    }

    switch (last - last_aligned) {
    case 7:
        lhs[6] = ~lhs[6];
        /* FALLTHRU */
    case 6:
        lhs[5] = ~lhs[5];
        /* FALLTHRU */
    case 5:
        lhs[4] = ~lhs[4];
        /* FALLTHRU */
    case 4:
        lhs[3] = ~lhs[3];
        /* FALLTHRU */
    case 3:
        lhs[2] = ~lhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] = ~lhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] = ~lhs[0];
    }
}

static inline bool bitset_test_all(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    for (; lhs != last_aligned; lhs += 8) {
        if (*(uint64_t *)(&lhs[6]) != (uint64_t)-1) {
            return false;
        }
        if (*(uint64_t *)(&lhs[4]) != (uint64_t)-1) {
            return false;
        }
        if (*(uint64_t *)(&lhs[2]) != (uint64_t)-1) {
            return false;
        }
        if (*(uint64_t *)(&lhs[0]) != (uint64_t)-1) {
            return false;
        }
    }

    switch (last - last_aligned) {
    case 7:
        if (lhs[6] != (uint32_t)-1) {
            return false;
        }
        /* FALLTHRU */
    case 6:
        if (lhs[5] != (uint32_t)-1) {
            return false;
        }
        /* FALLTHRU */
    case 5:
        if (lhs[4] != (uint32_t)-1) {
            return false;
        }
        /* FALLTHRU */
    case 4:
        if (lhs[3] != (uint32_t)-1) {
            return false;
        }
        /* FALLTHRU */
    case 3:
        if (lhs[2] != (uint32_t)-1) {
            return false;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != (uint32_t)-1) {
            return false;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != (uint32_t)-1) {
            return false;
        }
    }
    return true;
}

static inline bool bitset_test_any(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    for (; lhs != last_aligned; lhs += 8) {
        if (*(uint64_t *)(&lhs[6]) != 0u) {
            return true;
        }
        if (*(uint64_t *)(&lhs[4]) != 0u) {
            return true;
        }
        if (*(uint64_t *)(&lhs[2]) != 0u) {
            return true;
        }
        if (*(uint64_t *)(&lhs[0]) != 0u) {
            return true;
        }
    }

    switch (last - last_aligned) {
    case 7:
        if (lhs[6] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 6:
        if (lhs[5] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 5:
        if (lhs[4] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 4:
        if (lhs[3] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 3:
        if (lhs[2] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != 0u) {
            return true;
        }
    }
    return false;
}

static inline bool bitset_test_none(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 3) << 3);

    for (; lhs != last_aligned; lhs += 8) {
        if (*(uint64_t *)(&lhs[6]) != 0u) {
            return false;
        }
        if (*(uint64_t *)(&lhs[4]) != 0u) {
            return false;
        }
        if (*(uint64_t *)(&lhs[2]) != 0u) {
            return false;
        }
        if (*(uint64_t *)(&lhs[0]) != 0u) {
            return false;
        }
    }

    switch (last - last_aligned) {
    case 7:
        if (lhs[6] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 6:
        if (lhs[5] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 5:
        if (lhs[4] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 4:
        if (lhs[3] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 3:
        if (lhs[2] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != 0u) {
            return false;
        }
    }
    return true;
}

static inline size_t bitset_cardinality(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
    size_t count = 0;

    for (; lhs != last_aligned; lhs += 8) {
        count += bitset_popcount64(*(uint64_t *)(&lhs[6]));
        count += bitset_popcount64(*(uint64_t *)(&lhs[4]));
        count += bitset_popcount64(*(uint64_t *)(&lhs[2]));
        count += bitset_popcount64(*(uint64_t *)(&lhs[0]));
    }
    switch (last - last_aligned) {
    case 7:
        count += bitset_popcount32(lhs[6]);
        /* FALLTHRU */
    case 6:
        count += bitset_popcount32(lhs[5]);
        /* FALLTHRU */
    case 5:
        count += bitset_popcount32(lhs[4]);
        /* FALLTHRU */
    case 4:
        count += bitset_popcount32(lhs[3]);
        /* FALLTHRU */
    case 3:
        count += bitset_popcount32(lhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0]);
    }
    return count;
}

static inline size_t bitset_xor_cardinality(const uint32_t *lhs,
                                            const uint32_t *rhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
    size_t count = 0;

    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[6]) ^ *(uint64_t *)(&rhs[6]));
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[4]) ^ *(uint64_t *)(&rhs[4]));
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[2]) ^ *(uint64_t *)(&rhs[2]));
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[0]) ^ *(uint64_t *)(&rhs[0]));
    }
    switch (last - last_aligned) {
    case 7:
        count += bitset_popcount32(lhs[6] ^ rhs[6]);
        /* FALLTHRU */
    case 6:
        count += bitset_popcount32(lhs[5] ^ rhs[5]);
        /* FALLTHRU */
    case 5:
        count += bitset_popcount32(lhs[4] ^ rhs[4]);
        /* FALLTHRU */
    case 4:
        count += bitset_popcount32(lhs[3] ^ rhs[3]);
        /* FALLTHRU */
    case 3:
        count += bitset_popcount32(lhs[2] ^ rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] ^ rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] ^ rhs[0]);
    }
    return count;
}

static inline size_t bitset_and_cardinality(const uint32_t *lhs,
                                            const uint32_t *rhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
    size_t count = 0;

    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[6]) & *(uint64_t *)(&rhs[6]));
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[4]) & *(uint64_t *)(&rhs[4]));
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[2]) & *(uint64_t *)(&rhs[2]));
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[0]) & *(uint64_t *)(&rhs[0]));
    }
    switch (last - last_aligned) {
    case 7:
        count += bitset_popcount32(lhs[6] & rhs[6]);
        /* FALLTHRU */
    case 6:
        count += bitset_popcount32(lhs[5] & rhs[5]);
        /* FALLTHRU */
    case 5:
        count += bitset_popcount32(lhs[4] & rhs[4]);
        /* FALLTHRU */
    case 4:
        count += bitset_popcount32(lhs[3] & rhs[3]);
        /* FALLTHRU */
    case 3:
        count += bitset_popcount32(lhs[2] & rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] & rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] & rhs[0]);
    }
    return count;
}

static inline size_t bitset_andnot_cardinality(const uint32_t *lhs,
                                               const uint32_t *rhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
    size_t count = 0;

    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
        count += bitset_popcount64(*(uint64_t *)(&lhs[6]) &
                                   ~(*(uint64_t *)(&rhs[6])));
        count += bitset_popcount64(*(uint64_t *)(&lhs[4]) &
                                   ~(*(uint64_t *)(&rhs[4])));
        count += bitset_popcount64(*(uint64_t *)(&lhs[2]) &
                                   ~(*(uint64_t *)(&rhs[2])));
        count += bitset_popcount64(*(uint64_t *)(&lhs[0]) &
                                   ~(*(uint64_t *)(&rhs[0])));
    }
    switch (last - last_aligned) {
    case 7:
        count += bitset_popcount32(lhs[6] & ~rhs[6]);
        /* FALLTHRU */
    case 6:
        count += bitset_popcount32(lhs[5] & ~rhs[5]);
        /* FALLTHRU */
    case 5:
        count += bitset_popcount32(lhs[4] & ~rhs[4]);
        /* FALLTHRU */
    case 4:
        count += bitset_popcount32(lhs[3] & ~rhs[3]);
        /* FALLTHRU */
    case 3:
        count += bitset_popcount32(lhs[2] & ~rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] & ~rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] & ~rhs[0]);
    }
    return count;
}

static inline size_t bitset_or_cardinality(const uint32_t *lhs,
                                           const uint32_t *rhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
    size_t count = 0;

    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[6]) | *(uint64_t *)(&rhs[6]));
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[4]) | *(uint64_t *)(&rhs[4]));
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[2]) | *(uint64_t *)(&rhs[2]));
        count +=
            bitset_popcount64(*(uint64_t *)(&lhs[0]) | *(uint64_t *)(&rhs[0]));
    }
    switch (last - last_aligned) {
    case 7:
        count += bitset_popcount32(lhs[6] | rhs[6]);
        /* FALLTHRU */
    case 6:
        count += bitset_popcount32(lhs[5] | rhs[5]);
        /* FALLTHRU */
    case 5:
        count += bitset_popcount32(lhs[4] | rhs[4]);
        /* FALLTHRU */
    case 4:
        count += bitset_popcount32(lhs[3] | rhs[3]);
        /* FALLTHRU */
    case 3:
        count += bitset_popcount32(lhs[2] | rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] | rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] | rhs[0]);
    }
    return count;
}
#else  // PLATFORM_M64

static inline void bitset_and(uint32_t *lhs, const uint32_t *rhs, size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        lhs[3] &= rhs[3];
        lhs[2] &= rhs[2];
        lhs[1] &= rhs[1];
        lhs[0] &= rhs[0];
    }

    switch (last - last_aligned) {
    case 3:
        lhs[2] &= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] &= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] &= rhs[0];
    }
}

static inline void bitset_andnot(uint32_t *lhs, const uint32_t *rhs,
                                 size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        lhs[3] &= ~rhs[3];
        lhs[2] &= ~rhs[2];
        lhs[1] &= ~rhs[1];
        lhs[0] &= ~rhs[0];
    }

    switch (last - last_aligned) {
    case 3:
        lhs[2] &= ~rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] &= ~rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] &= ~rhs[0];
    }
}

static inline void bitset_or(uint32_t *lhs, const uint32_t *rhs, size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        lhs[3] |= rhs[3];
        lhs[2] |= rhs[2];
        lhs[1] |= rhs[1];
        lhs[0] |= rhs[0];
    }

    switch (last - last_aligned) {
    case 3:
        lhs[2] |= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] |= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] |= rhs[0];
    }
}

static inline void bitset_xor(uint32_t *lhs, const uint32_t *rhs, size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        lhs[3] ^= rhs[3];
        lhs[2] ^= rhs[2];
        lhs[1] ^= rhs[1];
        lhs[0] ^= rhs[0];
    }

    switch (last - last_aligned) {
    case 3:
        lhs[2] ^= rhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] ^= rhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] ^= rhs[0];
    }
}

static inline void bitset_not(uint32_t *lhs, size_t size)
{
    uint32_t *last = lhs + size;
    uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    for (; lhs != last_aligned; lhs += 4) {
        lhs[3] = ~lhs[3];
        lhs[2] = ~lhs[2];
        lhs[1] = ~lhs[1];
        lhs[0] = ~lhs[0];
    }

    switch (last - last_aligned) {
    case 3:
        lhs[2] = ~lhs[2];
        /* FALLTHRU */
    case 2:
        lhs[1] = ~lhs[1];
        /* FALLTHRU */
    case 1:
        lhs[0] = ~lhs[0];
    }
}

static inline bool bitset_test_all(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    for (; lhs != last_aligned; lhs += 4) {
        if (lhs[3] != (uint32_t)-1) {
            return false;
        }
        if (lhs[2] != (uint32_t)-1) {
            return false;
        }
        if (lhs[1] != (uint32_t)-1) {
            return false;
        }
        if (lhs[0] != (uint32_t)-1) {
            return false;
        }
    }

    switch (last - last_aligned) {
    case 3:
        if (lhs[2] != (uint32_t)-1) {
            return false;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != (uint32_t)-1) {
            return false;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != (uint32_t)-1) {
            return false;
        }
    }
    return true;
}

static inline bool bitset_test_any(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    for (; lhs != last_aligned; lhs += 4) {
        if (lhs[3] != 0u) {
            return true;
        }
        if (lhs[2] != 0u) {
            return true;
        }
        if (lhs[1] != 0u) {
            return true;
        }
        if (lhs[0] != 0u) {
            return true;
        }
    }

    switch (last - last_aligned) {
    case 3:
        if (lhs[2] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != 0u) {
            return true;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != 0u) {
            return true;
        }
    }
    return false;
}

static inline bool bitset_test_none(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

    for (; lhs != last_aligned; lhs += 4) {
        if (lhs[3] != 0u) {
            return false;
        }
        if (lhs[2] != 0u) {
            return false;
        }
        if (lhs[1] != 0u) {
            return false;
        }
        if (lhs[0] != 0u) {
            return false;
        }
    }

    switch (last - last_aligned) {
    case 3:
        if (lhs[2] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 2:
        if (lhs[1] != 0u) {
            return false;
        }
        /* FALLTHRU */
    case 1:
        if (lhs[0] != 0u) {
            return false;
        }
    }
    return true;
}

static inline size_t bitset_cardinality(const uint32_t *lhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
    size_t count = 0;

    for (; lhs != last_aligned; lhs += 4) {
        count += bitset_popcount32(lhs[3]);
        count += bitset_popcount32(lhs[2]);
        count += bitset_popcount32(lhs[1]);
        count += bitset_popcount32(lhs[0]);
    }
    switch (last - last_aligned) {
    case 3:
        count += bitset_popcount32(lhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0]);
    }
    return count;
}

static inline size_t bitset_xor_cardinality(const uint32_t *lhs,
                                            const uint32_t *rhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
    size_t count = 0;

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        count += bitset_popcount32(lhs[3] ^ rhs[3]);
        count += bitset_popcount32(lhs[2] ^ rhs[2]);
        count += bitset_popcount32(lhs[1] ^ rhs[1]);
        count += bitset_popcount32(lhs[0] ^ rhs[0]);
    }
    switch (last - last_aligned) {
    case 3:
        count += bitset_popcount32(lhs[2] ^ rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] ^ rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] ^ rhs[0]);
    }
    return count;
}

static inline size_t bitset_and_cardinality(const uint32_t *lhs,
                                            const uint32_t *rhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
    size_t count = 0;

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        count += bitset_popcount32(lhs[3] & rhs[3]);
        count += bitset_popcount32(lhs[2] & rhs[2]);
        count += bitset_popcount32(lhs[1] & rhs[1]);
        count += bitset_popcount32(lhs[0] & rhs[0]);
    }
    switch (last - last_aligned) {
    case 3:
        count += bitset_popcount32(lhs[2] & rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] & rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] & rhs[0]);
    }
    return count;
}

static inline size_t bitset_andnot_cardinality(const uint32_t *lhs,
                                               const uint32_t *rhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
    size_t count = 0;

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        count += bitset_popcount32(lhs[3] & ~rhs[3]);
        count += bitset_popcount32(lhs[2] & ~rhs[2]);
        count += bitset_popcount32(lhs[1] & ~rhs[1]);
        count += bitset_popcount32(lhs[0] & ~rhs[0]);
    }
    switch (last - last_aligned) {
    case 3:
        count += bitset_popcount32(lhs[2] & ~rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] & ~rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] & ~rhs[0]);
    }
    return count;
}

static inline size_t bitset_or_cardinality(const uint32_t *lhs,
                                           const uint32_t *rhs, size_t size)
{
    const uint32_t *last = lhs + size;
    const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
    size_t count = 0;

    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
        count += bitset_popcount32(lhs[3] | rhs[3]);
        count += bitset_popcount32(lhs[2] | rhs[2]);
        count += bitset_popcount32(lhs[1] | rhs[1]);
        count += bitset_popcount32(lhs[0] | rhs[0]);
    }
    switch (last - last_aligned) {
    case 3:
        count += bitset_popcount32(lhs[2] | rhs[2]);
        /* FALLTHRU */
    case 2:
        count += bitset_popcount32(lhs[1] | rhs[1]);
        /* FALLTHRU */
    case 1:
        count += bitset_popcount32(lhs[0] | rhs[0]);
    }
    return count;
}
#endif // PLATFORM_M64

namespace mercury {
namespace internal {

void BitsetAnd(uint32_t *lhs, const uint32_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (CpuFeatures::AVX2()) {
        bitset_and_v256(lhs, rhs, size);
        return;
    }
#endif // __AVX2__

#if defined(__SSE2__)
#if defined(__SSE3__)
    if (CpuFeatures::SSE3())
#else
    if (CpuFeatures::SSE2())
#endif // __SSE3__
    {
        bitset_and_v128(lhs, rhs, size);
        return;
    }
#endif // __SSE2__
    bitset_and(lhs, rhs, size);
}

void BitsetAndnot(uint32_t *lhs, const uint32_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (CpuFeatures::AVX2()) {
        bitset_andnot_v256(lhs, rhs, size);
        return;
    }
#endif // __AVX2__

#if defined(__SSE2__)
#if defined(__SSE3__)
    if (CpuFeatures::SSE3())
#else
    if (CpuFeatures::SSE2())
#endif // __SSE3__
    {
        bitset_andnot_v128(lhs, rhs, size);
        return;
    }
#endif // __SSE2__
    bitset_andnot(lhs, rhs, size);
}

void BitsetOr(uint32_t *lhs, const uint32_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (CpuFeatures::AVX2()) {
        bitset_or_v256(lhs, rhs, size);
        return;
    }
#endif // __AVX2__

#if defined(__SSE2__)
#if defined(__SSE3__)
    if (CpuFeatures::SSE3())
#else
    if (CpuFeatures::SSE2())
#endif // __SSE3__
    {
        bitset_or_v128(lhs, rhs, size);
        return;
    }
#endif // __SSE2__
    bitset_or(lhs, rhs, size);
}

void BitsetXor(uint32_t *lhs, const uint32_t *rhs, size_t size)
{
#if defined(__AVX2__)
    if (CpuFeatures::AVX2()) {
        bitset_xor_v256(lhs, rhs, size);
        return;
    }
#endif // __AVX2__

#if defined(__SSE2__)
#if defined(__SSE3__)
    if (CpuFeatures::SSE3())
#else
    if (CpuFeatures::SSE2())
#endif // __SSE3__
    {
        bitset_xor_v128(lhs, rhs, size);
        return;
    }
#endif // __SSE2__
    bitset_xor(lhs, rhs, size);
}

void BitsetNot(uint32_t *arr, size_t size)
{
#if defined(__AVX2__)
    if (CpuFeatures::AVX2()) {
        bitset_not_v256(arr, size);
        return;
    }
#endif // __AVX2__

#if defined(__SSE2__)
#if defined(__SSE3__)
    if (CpuFeatures::SSE3())
#else
    if (CpuFeatures::SSE2())
#endif // __SSE3__
    {
        bitset_not_v128(arr, size);
        return;
    }
#endif // __SSE2__
    bitset_not(arr, size);
}

bool BitsetTestAll(const uint32_t *arr, size_t size)
{
#if defined(__AVX2__)
    if (CpuFeatures::AVX2()) {
        return bitset_test_all_v256(arr, size);
    }
#endif // __AVX2__

#if defined(__SSE2__)
#if defined(__SSE4_1__)
    if (CpuFeatures::SSE4_1())
#elif defined(__SSE3__)
    if (CpuFeatures::SSE3())
#else
    if (CpuFeatures::SSE2())
#endif // __SSE4_1__
    {
        return bitset_test_all_v128(arr, size);
    }
#endif // __SSE2__
    return bitset_test_all(arr, size);
}

bool BitsetTestAny(const uint32_t *arr, size_t size)
{
#if defined(__AVX2__)
    if (CpuFeatures::AVX2()) {
        return bitset_test_any_v256(arr, size);
    }
#endif // __AVX2__

#if defined(__SSE2__)
#if defined(__SSE4_1__)
    if (CpuFeatures::SSE4_1())
#elif defined(__SSE3__)
    if (CpuFeatures::SSE3())
#else
    if (CpuFeatures::SSE2())
#endif // __SSE4_1__
    {
        return bitset_test_any_v128(arr, size);
    }
#endif // __SSE2__
    return bitset_test_any(arr, size);
}

bool BitsetTestNone(const uint32_t *arr, size_t size)
{
#if defined(__AVX2__)
    if (CpuFeatures::AVX2()) {
        return bitset_test_none_v256(arr, size);
    }
#endif // __AVX2__

#if defined(__SSE2__)
#if defined(__SSE4_1__)
    if (CpuFeatures::SSE4_1())
#elif defined(__SSE3__)
    if (CpuFeatures::SSE3())
#else
    if (CpuFeatures::SSE2())
#endif // __SSE4_1__
    {
        return bitset_test_none_v128(arr, size);
    }
#endif // __SSE2__
    return bitset_test_none(arr, size);
}

size_t BitsetCardinality(const uint32_t *arr, size_t size)
{
#if defined(__AVX2__)
    if (size > 31 && CpuFeatures::AVX2()) {
        return bitset_cardinality_v256(arr, size);
    }
#endif // __AVX2__
    return bitset_cardinality(arr, size);
}

size_t BitsetAndCardinality(const uint32_t *lhs, const uint32_t *rhs,
                            size_t size)
{
#if defined(__AVX2__)
    if (size > 31 && CpuFeatures::AVX2()) {
        return bitset_and_cardinality_v256(lhs, rhs, size);
    }
#endif // __AVX2__
    return bitset_and_cardinality(lhs, rhs, size);
}

size_t BitsetAndnotCardinality(const uint32_t *lhs, const uint32_t *rhs,
                               size_t size)
{
#if defined(__AVX2__)
    if (size > 31 && CpuFeatures::AVX2()) {
        return bitset_andnot_cardinality_v256(lhs, rhs, size);
    }
#endif // __AVX2__
    return bitset_andnot_cardinality(lhs, rhs, size);
}

size_t BitsetXorCardinality(const uint32_t *lhs, const uint32_t *rhs,
                            size_t size)
{
#if defined(__AVX2__)
    if (size > 31 && CpuFeatures::AVX2()) {
        return bitset_xor_cardinality_v256(lhs, rhs, size);
    }
#endif // __AVX2__
    return bitset_xor_cardinality(lhs, rhs, size);
}

size_t BitsetOrCardinality(const uint32_t *lhs, const uint32_t *rhs,
                           size_t size)
{
#if defined(__AVX2__)
    if (size > 31 && CpuFeatures::AVX2()) {
        return bitset_or_cardinality_v256(lhs, rhs, size);
    }
#endif // __AVX2__
    return bitset_or_cardinality(lhs, rhs, size);
}

void BitsetExtract(const uint32_t *arr, size_t size, size_t base,
                   std::vector<size_t> *out)
{
    const uint32_t *iter = arr;
    const uint32_t *last = arr + size;

    for (; iter != last; ++iter) {
        uint32_t w = *iter;

        while (w != 0) {
            uint32_t c = platform_ctz32(w);
            w &= ~(1u << c);
            out->push_back(base + c);
        }
        base += 32u;
    }
}

} // namespace internal
} // namespace mercury
