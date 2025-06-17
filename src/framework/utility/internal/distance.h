/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     distance.h
 *   \author   Hechong.xyf
 *   \date     Jan 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Distance (internal)
 */

#ifndef __MERCURY_UTILITY_INTERNAL_DISTANCE_H__
#define __MERCURY_UTILITY_INTERNAL_DISTANCE_H__

#include "bitset.h"
#include <cmath>

namespace mercury {
namespace internal {

//! Compute the hamming distance between two vectors
static inline float HammingDistance(const uint32_t *lhs, const uint32_t *rhs,
                                    size_t size)
{
    // M10 + M01
    return BitsetXorCardinality(lhs, rhs, size);
}

//! Compute the jaccard distance between two vectors
static inline float JaccardDistance(const uint32_t *lhs, const uint32_t *rhs,
                                    size_t size)
{
    // (M10 + M01) / (M11 + M10 + M01)
    size_t lower = BitsetOrCardinality(lhs, rhs, size);
    if (lower != 0) {
        size_t hamming = BitsetXorCardinality(lhs, rhs, size);
        return ((float)hamming / (float)lower);
    }
    return 0.0f;
}

//! Compute the matching distance between two vectors
static inline float MatchingDistance(const uint32_t *lhs, const uint32_t *rhs,
                                     size_t size)
{
    // 1 - (M11 + M00) / (M11 + M10 + M01 + M00) =
    // (M10 + M01) / N
    return ((float)(BitsetXorCardinality(lhs, rhs, size)) / (float)(size << 5));
}

//! Compute the dice distance between two vectors
static inline float DiceDistance(const uint32_t *lhs, const uint32_t *rhs,
                                 size_t size)
{
    // (M10 + M01) / (M11 * 2 + (M10 + M01))
    size_t hamming = BitsetXorCardinality(lhs, rhs, size);
    size_t overlap = BitsetAndCardinality(lhs, rhs, size);
    size_t lower = 2 * overlap + hamming;
    if (lower != 0) {
        return ((float)hamming / (float)lower);
    }
    return 0.0f;
}

//! Compute the rogers tanimoto distance between two vectors
static inline float RogersTanimotoDistance(const uint32_t *lhs,
                                           const uint32_t *rhs, size_t size)
{
    // (M10 + M01) * 2 / (M11 + (M10 + M01) * 2 + M00) =
    // (M10 + M01) * 2 / (N + M10 + M01)
    size_t hamming = BitsetXorCardinality(lhs, rhs, size);
    return ((float)(hamming * 2) / (float)((size << 5) + hamming));
}

//! Compute the russell rao distance between two vectors
static inline float RussellRaoDistance(const uint32_t *lhs, const uint32_t *rhs,
                                       size_t size)
{
    // (M10 + M01 + M00) / N = (N - M11) / N
    size_t total = (size << 5);
    return ((float)(total - BitsetAndCardinality(lhs, rhs, size)) /
            (float)total);
}

//! Compute the sokal michener distance between two vectors
static inline float SokalMichenerDistance(const uint32_t *lhs,
                                          const uint32_t *rhs, size_t size)
{
    return MatchingDistance(lhs, rhs, size);
}

//! Compute the sokal sneath distance between two vectors
static inline float SokalSneathDistance(const uint32_t *lhs,
                                        const uint32_t *rhs, size_t size)
{
    // 1.0 - (M11 / (M11 + (M10 + M01) * 2)) =
    // (M10 + M01) * 2 / (M11 + (M10 + M01) * 2)
    size_t hamming_x2 = BitsetXorCardinality(lhs, rhs, size) * 2;
    size_t overlap = BitsetAndCardinality(lhs, rhs, size);
    size_t lower = overlap + hamming_x2;
    if (lower != 0) {
        return ((float)hamming_x2 / (float)lower);
    }
    return 0.0f;
}

//! Compute the yule distance between two vectors
static inline float YuleDistance(const uint32_t *lhs, const uint32_t *rhs,
                                 size_t size)
{
    // 2 * M10 * M01 / (M11 * M00 + M10 * M01) =
    // 2 * M10 * M01 / (M11 * (N - M10 - M01 - M11) + M10 * M01)
    size_t m10 = BitsetAndnotCardinality(lhs, rhs, size);
    size_t m01 = BitsetAndnotCardinality(rhs, lhs, size);
    size_t m11 = BitsetAndCardinality(lhs, rhs, size);
    size_t m10_x_m01 = m10 * m01;
    return ((float)(2 * m10_x_m01) /
            (float)(m11 * ((size << 5) - m10 - m01 - m11) + m10_x_m01));
}

//! Compute the squared euclidean distance between two vectors
float SquaredEuclideanDistance(const float *lhs, const float *rhs, size_t size);

//! Compute the euclidean distance between two vectors
float EuclideanDistance(const float *lhs, const float *rhs, size_t size);

//! Compute the weighted squared euclidean distance between two vectors
float SquaredEuclideanDistance(const float *lhs, const float *rhs,
                               const float *wgt, size_t size);

//! Compute the weighted euclidean distance between two vectors
float EuclideanDistance(const float *lhs, const float *rhs, const float *wgt,
                        size_t size);

//! Compute the normalized squared euclidean distance between two vectors
float NormalizedSquaredEuclideanDistance(const float *lhs, const float *rhs,
                                         size_t size);

//! Compute the normalized euclidean distance between two vectors
float NormalizedEuclideanDistance(const float *lhs, const float *rhs,
                                  size_t size);

//! Compute the manhattan distance between two vectors
float ManhattanDistance(const float *lhs, const float *rhs, size_t size);

//! Compute the chebyshev distance between two vectors
float ChebyshevDistance(const float *lhs, const float *rhs, size_t size);

//! Compute the chessboard distance between two vectors
static inline float ChessboardDistance(const float *lhs, const float *rhs,
                                       size_t size)
{
    return ChebyshevDistance(lhs, rhs, size);
}

//! Compute the cosine distance between two vectors
float CosineDistance(const float *lhs, const float *rhs, size_t size);

//! Compute the canberra distance between two vectors
float CanberraDistance(const float *lhs, const float *rhs, size_t size);

//! Compute the bray-curtis distance between two vectors
float BrayCurtisDistance(const float *lhs, const float *rhs, size_t size);

//! Compute the correlation distance between two vectors
float CorrelationDistance(const float *lhs, const float *rhs, size_t size);

//! Compute the binary distance between two vectors
float BinaryDistance(const float *lhs, const float *rhs, size_t size);

//! Compute the inner product between two vectors
float InnerProduct(const float *lhs, const float *rhs, size_t size);

//! Compute the squared euclidean distance between two vectors
float SquaredEuclideanDistance(const int16_t *lhs, const int16_t *rhs,
                               size_t size);

//! Compute the euclidean distance between two vectors
float EuclideanDistance(const int16_t *lhs, const int16_t *rhs, size_t size);

//! Compute the weighted squared euclidean distance between two vectors
float SquaredEuclideanDistance(const int16_t *lhs, const int16_t *rhs,
                               const float *wgt, size_t size);

//! Compute the weighted euclidean distance between two vectors
float EuclideanDistance(const int16_t *lhs, const int16_t *rhs,
                        const float *wgt, size_t size);

//! Compute the normalized squared euclidean distance between two vectors
float NormalizedSquaredEuclideanDistance(const int16_t *lhs, const int16_t *rhs,
                                         size_t size);

//! Compute the normalized euclidean distance between two vectors
float NormalizedEuclideanDistance(const int16_t *lhs, const int16_t *rhs,
                                  size_t size);

//! Compute the manhattan distance between two vectors
float ManhattanDistance(const int16_t *lhs, const int16_t *rhs, size_t size);

//! Compute the chebyshev distance between two vectors
float ChebyshevDistance(const int16_t *lhs, const int16_t *rhs, size_t size);

//! Compute the chessboard distance between two vectors
static inline float ChessboardDistance(const int16_t *lhs, const int16_t *rhs,
                                       size_t size)
{
    return ChebyshevDistance(lhs, rhs, size);
}

//! Compute the cosine distance between two vectors
float CosineDistance(const int16_t *lhs, const int16_t *rhs, size_t size);

//! Compute the canberra distance between two vectors
float CanberraDistance(const int16_t *lhs, const int16_t *rhs, size_t size);

//! Compute the bray-curtis distance between two vectors
float BrayCurtisDistance(const int16_t *lhs, const int16_t *rhs, size_t size);

//! Compute the correlation distance between two vectors
float CorrelationDistance(const int16_t *lhs, const int16_t *rhs, size_t size);

//! Compute the binary distance between two vectors
float BinaryDistance(const int16_t *lhs, const int16_t *rhs, size_t size);

//! Compute the inner product between two vectors
float InnerProduct(const int16_t *lhs, const int16_t *rhs, size_t size);

//! Compute the squared euclidean distance between two vectors
float SquaredEuclideanDistance(const int8_t *lhs, const int8_t *rhs,
                               size_t size);

//! Compute the euclidean distance between two vectors
float EuclideanDistance(const int8_t *lhs, const int8_t *rhs, size_t size);

//! Compute the weighted squared euclidean distance between two vectors
float SquaredEuclideanDistance(const int8_t *lhs, const int8_t *rhs,
                               const float *wgt, size_t size);

//! Compute the weighted euclidean distance between two vectors
float EuclideanDistance(const int8_t *lhs, const int8_t *rhs, const float *wgt,
                        size_t size);

//! Compute the normalized squared euclidean distance between two vectors
float NormalizedSquaredEuclideanDistance(const int8_t *lhs, const int8_t *rhs,
                                         size_t size);

//! Compute the normalized euclidean distance between two vectors
float NormalizedEuclideanDistance(const int8_t *lhs, const int8_t *rhs,
                                  size_t size);

//! Compute the manhattan distance between two vectors
float ManhattanDistance(const int8_t *lhs, const int8_t *rhs, size_t size);

//! Compute the chebyshev distance between two vectors
float ChebyshevDistance(const int8_t *lhs, const int8_t *rhs, size_t size);

//! Compute the chessboard distance between two vectors
static inline float ChessboardDistance(const int8_t *lhs, const int8_t *rhs,
                                       size_t size)
{
    return ChebyshevDistance(lhs, rhs, size);
}

//! Compute the cosine distance between two vectors
float CosineDistance(const int8_t *lhs, const int8_t *rhs, size_t size);

//! Compute the canberra distance between two vectors
float CanberraDistance(const int8_t *lhs, const int8_t *rhs, size_t size);

//! Compute the bray-curtis distance between two vectors
float BrayCurtisDistance(const int8_t *lhs, const int8_t *rhs, size_t size);

//! Compute the correlation distance between two vectors
float CorrelationDistance(const int8_t *lhs, const int8_t *rhs, size_t size);

//! Compute the binary distance between two vectors
float BinaryDistance(const int8_t *lhs, const int8_t *rhs, size_t size);

//! Compute the inner product between two vectors
float InnerProduct(const int8_t *lhs, const int8_t *rhs, size_t size);

} // namespace internal
} // namespace mercury

#endif // __MERCURY_UTILITY_INTERNAL_DISTANCE_H__
