/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     bitset.h
 *   \author   Hechong.xyf
 *   \date     Dec 2017
 *   \version  1.0.0
 *   \brief    Interface of Bitset Utility
 */

#ifndef __MERCURY_UTILITY_INTERNAL_BITSET_H__
#define __MERCURY_UTILITY_INTERNAL_BITSET_H__

#include "platform.h"
#include <vector>

namespace mercury {
namespace internal {

//! Perform binary AND
void BitsetAnd(uint32_t *lhs, const uint32_t *rhs, size_t size);

//! Perform binary AND_NOT
void BitsetAndnot(uint32_t *lhs, const uint32_t *rhs, size_t size);

//! Perform binary OR
void BitsetOr(uint32_t *lhs, const uint32_t *rhs, size_t size);

//! Perform binary XOR
void BitsetXor(uint32_t *lhs, const uint32_t *rhs, size_t size);

//! Perform binary NOT
void BitsetNot(uint32_t *arr, size_t size);

//! Check if all bits are set to true
bool BitsetTestAll(const uint32_t *arr, size_t size);

//! Check if cube bits are set to true
bool BitsetTestAny(const uint32_t *arr, size_t size);

//! Check if none of the bits are set to true
bool BitsetTestNone(const uint32_t *arr, size_t size);

//! Compute the cardinality of a bitset
size_t BitsetCardinality(const uint32_t *arr, size_t size);

//! Compute the and cardinality distance between two bitsets
size_t BitsetAndCardinality(const uint32_t *lhs, const uint32_t *rhs,
                            size_t size);

//! Compute the andnot cardinality between two bitsets
size_t BitsetAndnotCardinality(const uint32_t *lhs, const uint32_t *rhs,
                               size_t size);

//! Compute the xor cardinality distance between two bitsets
size_t BitsetXorCardinality(const uint32_t *lhs, const uint32_t *rhs,
                            size_t size);

//! Compute the or cardinality between two bitsets
size_t BitsetOrCardinality(const uint32_t *lhs, const uint32_t *rhs,
                           size_t size);

//! Extract the bitset to an array
void BitsetExtract(const uint32_t *arr, size_t size, size_t base,
                   std::vector<size_t> *out);

} // namespace internal
} // namespace mercury

#endif // __MERCURY_UTILITY_INTERNAL_BITSET_H__
