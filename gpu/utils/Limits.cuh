/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "Pair.cuh"
#include <limits>

namespace proxima { namespace gpu {

template <typename T>
struct Limits {
};

// Unfortunately we can't use constexpr because there is no
// constexpr constructor for half
// FIXME: faiss CPU uses +/-FLT_MAX instead of +/-infinity
constexpr float kFloatMax = std::numeric_limits<float>::max();
constexpr float kFloatMin = std::numeric_limits<float>::lowest();

template <>
struct Limits<float> {
  static __device__ __host__ inline float getMin() {
    return kFloatMin;
  }
  static __device__ __host__ inline float getMax() {
    return kFloatMax;
  }
};

constexpr int kIntMax = std::numeric_limits<int>::max();
constexpr int kIntMin = std::numeric_limits<int>::lowest();

template <>
struct Limits<int> {
  static __device__ __host__ inline int getMin() {
    return kIntMin;
  }
  static __device__ __host__ inline int getMax() {
    return kIntMax;
  }
};

template<typename K, typename V>
struct Limits<Pair<K, V>> {
  static __device__ __host__ inline Pair<K, V> getMin() {
    return Pair<K, V>(Limits<K>::getMin(), Limits<V>::getMin());
  }

  static __device__ __host__ inline Pair<K, V> getMax() {
    return Pair<K, V>(Limits<K>::getMax(), Limits<V>::getMax());
  }
};

} } // namespace
