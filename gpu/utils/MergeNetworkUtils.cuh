/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

namespace proxima { namespace gpu {

template <typename T>
inline __device__ void swap(bool swap, T& x, T& y) {
  T tmp = x;
  x = swap ? y : x;
  y = swap ? tmp : y;
}

template <typename T>
inline __device__ void assign(bool assign, T& x, T y) {
  x = assign ? y : x;
}

} } // namespace
