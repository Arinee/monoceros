/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <cuda.h>
#include "DeviceDefs.cuh"

namespace proxima { namespace gpu {

template <typename T>
inline __device__ T shfl(const T val,
                         int srcLane, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_sync(0xffffffff, val, srcLane, width);
#else
  return __shfl(val, srcLane, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl(T* const val,
                         int srcLane, int width = kWarpSize) {
  static_assert(sizeof(T*) == sizeof(long long), "pointer size");
  long long v = (long long) val;

  return (T*) shfl(v, srcLane, width);
}

template <typename T>
inline __device__ T shfl_up(const T val,
                            unsigned int delta, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_up_sync(0xffffffff, val, delta, width);
#else
  return __shfl_up(val, delta, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_up(T* const val,
                             unsigned int delta, int width = kWarpSize) {
  static_assert(sizeof(T*) == sizeof(long long), "pointer size");
  long long v = (long long) val;

  return (T*) shfl_up(v, delta, width);
}

template <typename T>
inline __device__ T shfl_down(const T val,
                              unsigned int delta, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(0xffffffff, val, delta, width);
#else
  return __shfl_down(val, delta, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_down(T* const val,
                              unsigned int delta, int width = kWarpSize) {
  static_assert(sizeof(T*) == sizeof(long long), "pointer size");
  long long v = (long long) val;
  return (T*) shfl_down(v, delta, width);
}

template <typename T>
inline __device__ T shfl_xor(const T val,
                             int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_xor(T* const val,
                              int laneMask, int width = kWarpSize) {
  static_assert(sizeof(T*) == sizeof(long long), "pointer size");
  long long v = (long long) val;
  return (T*) shfl_xor(v, laneMask, width);
}

} } // namespace
