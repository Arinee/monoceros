/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     platform.h
 *   \author   Hechong.xyf
 *   \date     Dec 2017
 *   \version  1.0.0
 *   \brief    Interface of Platform Definition
 */

#ifndef __MERCURY_UTILITY_INTERNAL_PLATFORM_H__
#define __MERCURY_UTILITY_INTERNAL_PLATFORM_H__

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <unistd.h>
#include <x86intrin.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#ifndef NDEBUG
#define PLATFORM_DEBUG
#endif

//! Fixed Intel intrinsics macro in MSVC
#if defined(_MSC_VER)
#if _M_IX86_FP == 2
#define __SSE__ 1
#define __SSE2__ 1
#if _MSC_VER >= 1500
#define __SSE3__ 1
#define __SSSE3__ 1
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#endif
#elif _M_IX86_FP == 1
#define __SSE__ 1
#endif
#endif // _MSC_VER

#if defined(_WIN32) || defined(_WIN64)
#if defined(_WIN64)
#define PLATFORM_M64
#else
#define PLATFORM_M32
#endif
#endif

#if defined(__GNUC__)
#if defined(__x86_64__) || defined(__ppc64__)
#define PLATFORM_M64
#else
#define PLATFORM_M32
#endif
#endif

#ifndef PLATFORM_ALIGNED
#if defined(_MSC_VER)
#define PLATFORM_ALIGNED(x) __declspec(align(x))
#elif defined(__GNUC__)
#define PLATFORM_ALIGNED(x) __attribute__((aligned(x)))
#else
#define PLATFORM_ALIGNED(x)
#endif
#endif

//! Add 'inline' for MSVC
#if defined(_MSC_VER) && !defined(__cplusplus)
#if !defined(inline)
#define inline __inline
#endif
#endif

//! Add 'ssize_t' for MSVC
#if defined(_MSC_VER)
typedef intptr_t ssize_t;
#endif

#if defined(_MSC_VER)
//! Returns the number of trailing 0-bits in x
static inline int platform_ctz32(uint32_t x)
{
    unsigned long r = 0;
    _BitScanForward(&r, x);
    return (int)r;
}

//! Returns the number of leading 0-bits in x
static inline int platform_clz32(uint32_t x)
{
    unsigned long r = 0;
    _BitScanReverse(&r, x);
    return (31 - (int)r);
}

#if defined(PLATFORM_M64)
//! Returns the number of trailing 0-bits in x
static inline int platform_ctz64(uint64_t x)
{
    unsigned long r = 0;
    _BitScanForward64(&r, x);
    return (int)r;
}

//! Returns the number of leading 0-bits in x
static inline int platform_clz64(uint64_t x)
{
    unsigned long r = 0;
    _BitScanReverse64(&r, x);
    return (63 - (int)r);
}
#else
//! Returns the number of trailing 0-bits in x
static inline int platform_ctz64(uint64_t x)
{
    unsigned long r = 0;
    unsigned long m = (unsigned long)x;
    _BitScanForward(&r, m);
    if (r == 0) {
        m = (unsigned long)(x >> 32);
        _BitScanForward(&r, m);
        if (r != 0) {
            r += 32;
        }
    }
    return (int)r;
}

//! Returns the number of leading 0-bits in x
static inline int platform_clz64(uint64_t x)
{
    unsigned long r = 0;
    unsigned long m = (unsigned long)(x >> 32);
    _BitScanReverse(&r, m);
    if (r != 0) {
        return (31 - (int)r);
    }
    m = (unsigned long)x;
    _BitScanReverse(&r, m);
    return (63 - (int)r);
}
#endif // PLATFORM_M64

//! Counts the number of one bits
#define platform_popcount32(x) (__popcnt(x))
#define platform_popcount64(x) (__popcnt64(x))
#define platform_likely(x) (x)
#define platform_unlikely(x) (x)
#else // !_MSC_VER
#define platform_ctz32(x) (__builtin_ctz(x))
#define platform_ctz64(x) (__builtin_ctzll(x))
#define platform_clz32(x) (__builtin_clz(x))
#define platform_clz64(x) (__builtin_clzll(x))
#define platform_popcount32(x) (__builtin_popcount(x))
#define platform_popcount64(x) (__builtin_popcountl(x))
#define platform_likely(x) __builtin_expect(!!(x), 1)
#define platform_unlikely(x) __builtin_expect(!!(x), 0)
#endif // _MSC_VER

#if defined(_MSC_VER)
#define platform_aligned_malloc(SIZE, ALIGN)                                   \
    _aligned_malloc((size_t)(SIZE), (ALIGN))
#define platform_aligned_free _aligned_free
#else // !_MSC_VER
#if defined(_ISOC11_SOURCE)
#define platform_aligned_malloc(SIZE, ALIGN)                                   \
    aligned_alloc((ALIGN), (size_t)(SIZE))
#else // !_ISOC11_SOURCE
#define platform_aligned_malloc(SIZE, ALIGN)                                   \
    platform_posix_malloc((size_t)(SIZE), (ALIGN))
#endif // _ISOC11_SOURCE
#define platform_aligned_free free
#endif // _MSC_VER

#if defined(__AVX__) && (!defined(platform_malloc))
#define platform_malloc(SIZE) platform_aligned_malloc((SIZE), 32)
#elif defined(__SSE__) && (!defined(platform_malloc))
#define platform_malloc(SIZE) platform_aligned_malloc((SIZE), 16)
#endif

#if (defined(__AVX__) || defined(__SSE__)) && (!defined(platform_free))
#define platform_free platform_aligned_free
#endif

#ifndef platform_malloc
#define platform_malloc(SIZE) malloc((size_t)(SIZE))
#endif
#ifndef platform_free
#define platform_free free
#endif

#ifndef platform_align
#define platform_align(SIZE, BOUND) (((SIZE) + ((BOUND)-1)) & ~((BOUND)-1))
#endif

#ifndef platform_align8
#define platform_align8(SIZE) platform_align(SIZE, 8)
#endif

#ifndef platform_min
#define platform_min(A, B) (((A) < (B)) ? (A) : (B))
#endif

#ifndef platform_max
#define platform_max(A, B) (((A) > (B)) ? (A) : (B))
#endif

#ifndef platform_malloc_object
#define platform_malloc_object(TYPE) ((TYPE *)platform_malloc(sizeof(TYPE)))
#endif
#ifndef platform_malloc_array
#define platform_malloc_array(TYPE, SIZE)                                      \
    ((TYPE *)platform_malloc(SIZE * sizeof(TYPE)))
#endif

#ifndef platform_minus_if_ne_zero
#define platform_minus_if_ne_zero(COND)                                        \
    if (platform_unlikely((COND) != 0)) return (-1)
#endif

#ifndef platform_zero_if_ne_zero
#define platform_zero_if_ne_zero(COND)                                         \
    if (platform_unlikely((COND) != 0)) return (0)
#endif

#ifndef platform_null_if_ne_zero
#define platform_null_if_ne_zero(COND)                                         \
    if (platform_unlikely((COND) != 0)) return (NULL)
#endif

#ifndef platform_false_if_ne_zero
#define platform_false_if_ne_zero(COND)                                        \
    if (platform_unlikely((COND) != 0)) return (false)
#endif

#ifndef platform_return_if_ne_zero
#define platform_return_if_ne_zero(COND)                                       \
    if (platform_unlikely((COND) != 0)) return
#endif

#ifndef platform_break_if_ne_zero
#define platform_break_if_ne_zero(COND)                                        \
    if (platform_unlikely((COND) != 0)) break
#endif

#ifndef platform_continue_if_ne_zero
#define platform_continue_if_ne_zero(COND)                                     \
    if (platform_unlikely((COND) != 0)) continue
#endif

#ifndef platform_do_if_ne_zero
#define platform_do_if_ne_zero(COND) if (platform_unlikely((COND) != 0))
#endif

#ifndef platform_minus_if_lt_zero
#define platform_minus_if_lt_zero(COND)                                        \
    if (platform_unlikely((COND) < 0)) return (-1)
#endif

#ifndef platform_zero_if_lt_zero
#define platform_zero_if_lt_zero(COND)                                         \
    if (platform_unlikely((COND) < 0)) return (0)
#endif

#ifndef platform_null_if_lt_zero
#define platform_null_if_lt_zero(COND)                                         \
    if (platform_unlikely((COND) < 0)) return (NULL)
#endif

#ifndef platform_false_if_lt_zero
#define platform_false_if_lt_zero(COND)                                        \
    if (platform_unlikely((COND) < 0)) return (false)
#endif

#ifndef platform_return_if_lt_zero
#define platform_return_if_lt_zero(COND)                                       \
    if (platform_unlikely((COND) < 0)) return
#endif

#ifndef platform_break_if_lt_zero
#define platform_break_if_lt_zero(COND)                                        \
    if (platform_unlikely((COND) < 0)) break
#endif

#ifndef platform_continue_if_lt_zero
#define platform_continue_if_lt_zero(COND)                                     \
    if (platform_unlikely((COND) < 0)) continue
#endif

#ifndef platform_do_if_lt_zero
#define platform_do_if_lt_zero(COND) if (platform_unlikely((COND) < 0))
#endif

#ifndef platform_minus_if_false
#define platform_minus_if_false(COND)                                          \
    if (platform_unlikely(!(COND))) return (-1)
#endif

#ifndef platform_zero_if_false
#define platform_zero_if_false(COND)                                           \
    if (platform_unlikely(!(COND))) return (0)
#endif

#ifndef platform_null_if_false
#define platform_null_if_false(COND)                                           \
    if (platform_unlikely(!(COND))) return (NULL)
#endif

#ifndef platform_false_if_false
#define platform_false_if_false(COND)                                          \
    if (platform_unlikely(!(COND))) return (false)
#endif

#ifndef platform_return_if_false
#define platform_return_if_false(COND)                                         \
    if (platform_unlikely(!(COND))) return
#endif

#ifndef platform_break_if_false
#define platform_break_if_false(COND)                                          \
    if (platform_unlikely(!(COND))) break
#endif

#ifndef platform_continue_if_false
#define platform_continue_if_false(COND)                                       \
    if (platform_unlikely(!(COND))) continue
#endif

#ifndef platform_do_if_false
#define platform_do_if_false(COND) if (platform_unlikely(!(COND)))
#endif

#ifndef platform_compile_assert
#define platform_compile_assert(COND, MSG)                                     \
    typedef char Static_Assertion_##MSG[(!!(COND)) * 2 - 1]
#endif

#ifndef platform_static_assert3
#define platform_static_assert3(COND, LINE)                                    \
    platform_compile_assert(COND, At_Line_##LINE)
#endif

#ifndef platform_static_assert2
#define platform_static_assert2(COND, LINE) platform_static_assert3(COND, LINE)
#endif

#ifndef platform_static_assert
#define platform_static_assert(COND) platform_static_assert2(COND, __LINE__)
#endif

//! Abort and report if an assertion is failed
#ifndef platform_assert_abort
#define platform_assert_abort(COND, MSG)                                       \
    (void)(platform_likely(COND) ||                                            \
           (platform_assert_report(__FILE__, __FUNCTION__, __LINE__, #COND,    \
                                   (MSG)),                                     \
            abort(), 0))
#endif

#ifdef PLATFORM_DEBUG
#ifndef platform_assert
#define platform_assert(COND) platform_assert_abort(COND, "")
#endif
#ifndef platform_assert_with
#define platform_assert_with(COND, MSG) platform_assert_abort(COND, MSG)
#endif
#else // !PLATFORM_DEBUG
#ifndef platform_assert
#define platform_assert(COND) ((void)0)
#endif
#ifndef platform_assert_with
#define platform_assert_with(COND, MSG) ((void)0)
#endif
#endif // PLATFORM_DEBUG

#ifndef platform_check
#define platform_check(COND) platform_assert_abort(COND, "")
#endif
#ifndef platform_check_with
#define platform_check_with(COND, MSG) platform_assert_abort(COND, MSG)
#endif

#ifndef _MSC_VER
//! Allocates memory on a specified alignment boundary
static inline void *platform_posix_malloc(size_t size, size_t align)
{
    void *ptr;
    platform_null_if_ne_zero(posix_memalign(&ptr, align, size));
    return ptr;
}
#endif

//! Report an assertion is failed
static inline void platform_assert_report(const char *file, const char *func,
                                          int line, const char *cond,
                                          const char *msg)
{
    fprintf(stderr, "Assertion failed: (%s) in %s(), %s line %d. %s\n", cond,
            func, file, line, msg);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // __MERCURY_UTILITY_INTERNAL_PLATFORM_H__
