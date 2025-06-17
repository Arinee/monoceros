/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     distance.h
 *   \author   Hechong.xyf
 *   \date     Jan 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Distance
 */

#ifndef __MERCURY_UTILITY_DISTANCE_H__
#define __MERCURY_UTILITY_DISTANCE_H__

#include "fixed_bitset.h"
#include "fixed_vector.h"
#include "internal/distance.h"

namespace mercury {

/*! Distance module
 */
struct Distance
{
    //! Compute the hamming distance between two vectors
    template <size_t N>
    static float Hamming(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs)
    {
        return internal::HammingDistance(lhs.data(), rhs.data(),
                                         ((N + 0x1f) >> 5));
    }

    //! Compute the jaccard dissimilarity between two vectors
    template <size_t N>
    static float Jaccard(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs)
    {
        return internal::JaccardDistance(lhs.data(), rhs.data(),
                                         ((N + 0x1f) >> 5));
    }

    //! Compute the matching dissimilarity between two vectors
    template <size_t N>
    static float Matching(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs)
    {
        return internal::MatchingDistance(lhs.data(), rhs.data(),
                                          ((N + 0x1f) >> 5));
    }

    //! Compute the dice dissimilarity between two vectors
    template <size_t N>
    static float Dice(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs)
    {
        return internal::DiceDistance(lhs.data(), rhs.data(),
                                      ((N + 0x1f) >> 5));
    }

    //! Compute the rogers tanimoto dissimilarity between two vectors
    template <size_t N>
    static float RogersTanimoto(const FixedBitset<N> &lhs,
                                const FixedBitset<N> &rhs)
    {
        return internal::RogersTanimotoDistance(lhs.data(), rhs.data(),
                                                ((N + 0x1f) >> 5));
    }

    //! Compute the russell rao dissimilarity between two vectors
    template <size_t N>
    static float RussellRao(const FixedBitset<N> &lhs,
                            const FixedBitset<N> &rhs)
    {
        return internal::RussellRaoDistance(lhs.data(), rhs.data(),
                                            ((N + 0x1f) >> 5));
    }

    //! Compute the sokal michener dissimilarity between two vectors
    template <size_t N>
    static float SokalMichener(const FixedBitset<N> &lhs,
                               const FixedBitset<N> &rhs)
    {
        return internal::SokalMichenerDistance(lhs.data(), rhs.data(),
                                               ((N + 0x1f) >> 5));
    }

    //! Compute the sokal sneath dissimilarity between two vectors
    template <size_t N>
    static float SokalSneath(const FixedBitset<N> &lhs,
                             const FixedBitset<N> &rhs)
    {
        return internal::SokalSneathDistance(lhs.data(), rhs.data(),
                                             ((N + 0x1f) >> 5));
    }

    //! Compute the yule dissimilarity between two vectors
    template <size_t N>
    static float Yule(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs)
    {
        return internal::YuleDistance(lhs.data(), rhs.data(),
                                      ((N + 0x1f) >> 5));
    }

    //! Compute the euclidean distance between two vectors
    template <size_t N>
    static float Euclidean(const FloatFixedVector<N> &lhs,
                           const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::EuclideanDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the squared euclidean distance between two vectors
    template <size_t N>
    static float SquaredEuclidean(const FloatFixedVector<N> &lhs,
                                  const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::SquaredEuclideanDistance(lhs.data(),
                                                           rhs.data(), N);
    }

    //! Compute the weighted euclidean distance between two vectors
    template <size_t N>
    static float Euclidean(const FloatFixedVector<N> &lhs,
                           const FloatFixedVector<N> &rhs,
                           const FloatFixedVector<N> &wgt)
    {
        return mercury::internal::EuclideanDistance(lhs.data(), rhs.data(),
                                                    wgt.data(), N);
    }

    //! Compute the weighted squared euclidean distance between two vectors
    template <size_t N>
    static float SquaredEuclidean(const FloatFixedVector<N> &lhs,
                                  const FloatFixedVector<N> &rhs,
                                  const FloatFixedVector<N> &wgt)
    {
        return mercury::internal::SquaredEuclideanDistance(
            lhs.data(), rhs.data(), wgt.data(), N);
    }

    //! Compute the normalized euclidean distance between two vectors
    template <size_t N>
    static float NormalizedEuclidean(const FloatFixedVector<N> &lhs,
                                     const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::NormalizedEuclideanDistance(lhs.data(),
                                                              rhs.data(), N);
    }

    //! Compute the normalized squared euclidean distance between two vectors
    template <size_t N>
    static float NormalizedSquaredEuclidean(const FloatFixedVector<N> &lhs,
                                            const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::NormalizedSquaredEuclideanDistance(
            lhs.data(), rhs.data(), N);
    }

    //! Compute the manhattan distance between two vectors
    template <size_t N>
    static float Manhattan(const FloatFixedVector<N> &lhs,
                           const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::ManhattanDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the chebyshev distance between two vectors
    template <size_t N>
    static float Chebyshev(const FloatFixedVector<N> &lhs,
                           const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::ChebyshevDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the chessboard distance between two vectors
    template <size_t N>
    static float Chessboard(const FloatFixedVector<N> &lhs,
                            const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::ChessboardDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the cosine distance between two vectors
    template <size_t N>
    static float Cosine(const FloatFixedVector<N> &lhs,
                        const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::CosineDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the inner product between two vectors
    template <size_t N>
    static float InnerProduct(const FloatFixedVector<N> &lhs,
                              const FloatFixedVector<N> &rhs)
    {
        return -mercury::internal::InnerProduct(lhs.data(), rhs.data(), N);
    }

    //! Compute the canberra distance between two vectors
    template <size_t N>
    static float Canberra(const FloatFixedVector<N> &lhs,
                          const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::CanberraDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the bray-curtis distance between two vectors
    template <size_t N>
    static float BrayCurtis(const FloatFixedVector<N> &lhs,
                            const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::BrayCurtisDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the correlation distance between two vectors
    template <size_t N>
    static float Correlation(const FloatFixedVector<N> &lhs,
                             const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::CorrelationDistance(lhs.data(), rhs.data(),
                                                      N);
    }

    //! Compute the binary distance between two vectors
    template <size_t N>
    static float Binary(const FloatFixedVector<N> &lhs,
                        const FloatFixedVector<N> &rhs)
    {
        return mercury::internal::BinaryDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the euclidean distance between two vectors
    template <size_t N>
    static float Euclidean(const Int16FixedVector<N> &lhs,
                           const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::EuclideanDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the squared euclidean distance between two vectors
    template <size_t N>
    static float SquaredEuclidean(const Int16FixedVector<N> &lhs,
                                  const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::SquaredEuclideanDistance(lhs.data(),
                                                           rhs.data(), N);
    }

    //! Compute the weighted euclidean distance between two vectors
    template <size_t N>
    static float Euclidean(const Int16FixedVector<N> &lhs,
                           const Int16FixedVector<N> &rhs,
                           const FloatFixedVector<N> &wgt)
    {
        return mercury::internal::EuclideanDistance(lhs.data(), rhs.data(),
                                                    wgt.data(), N);
    }

    //! Compute the weighted squared euclidean distance between two vectors
    template <size_t N>
    static float SquaredEuclidean(const Int16FixedVector<N> &lhs,
                                  const Int16FixedVector<N> &rhs,
                                  const FloatFixedVector<N> &wgt)
    {
        return mercury::internal::SquaredEuclideanDistance(
            lhs.data(), rhs.data(), wgt.data(), N);
    }

    //! Compute the normalized euclidean distance between two vectors
    template <size_t N>
    static float NormalizedEuclidean(const Int16FixedVector<N> &lhs,
                                     const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::NormalizedEuclideanDistance(lhs.data(),
                                                              rhs.data(), N);
    }

    //! Compute the normalized squared euclidean distance between two vectors
    template <size_t N>
    static float NormalizedSquaredEuclidean(const Int16FixedVector<N> &lhs,
                                            const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::NormalizedSquaredEuclideanDistance(
            lhs.data(), rhs.data(), N);
    }

    //! Compute the manhattan distance between two vectors
    template <size_t N>
    static float Manhattan(const Int16FixedVector<N> &lhs,
                           const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::ManhattanDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the chebyshev distance between two vectors
    template <size_t N>
    static float Chebyshev(const Int16FixedVector<N> &lhs,
                           const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::ChebyshevDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the chessboard distance between two vectors
    template <size_t N>
    static float Chessboard(const Int16FixedVector<N> &lhs,
                            const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::ChessboardDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the cosine distance between two vectors
    template <size_t N>
    static float Cosine(const Int16FixedVector<N> &lhs,
                        const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::CosineDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the inner product between two vectors
    template <size_t N>
    static float InnerProduct(const Int16FixedVector<N> &lhs,
                              const Int16FixedVector<N> &rhs)
    {
        return -mercury::internal::InnerProduct(lhs.data(), rhs.data(), N);
    }

    //! Compute the canberra distance between two vectors
    template <size_t N>
    static float Canberra(const Int16FixedVector<N> &lhs,
                          const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::CanberraDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the bray-curtis distance between two vectors
    template <size_t N>
    static float BrayCurtis(const Int16FixedVector<N> &lhs,
                            const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::BrayCurtisDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the correlation distance between two vectors
    template <size_t N>
    static float Correlation(const Int16FixedVector<N> &lhs,
                             const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::CorrelationDistance(lhs.data(), rhs.data(),
                                                      N);
    }

    //! Compute the binary distance between two vectors
    template <size_t N>
    static float Binary(const Int16FixedVector<N> &lhs,
                        const Int16FixedVector<N> &rhs)
    {
        return mercury::internal::BinaryDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the euclidean distance between two vectors
    template <size_t N>
    static float Euclidean(const Int8FixedVector<N> &lhs,
                           const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::EuclideanDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the squared euclidean distance between two vectors
    template <size_t N>
    static float SquaredEuclidean(const Int8FixedVector<N> &lhs,
                                  const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::SquaredEuclideanDistance(lhs.data(),
                                                           rhs.data(), N);
    }

    //! Compute the weighted euclidean distance between two vectors
    template <size_t N>
    static float Euclidean(const Int8FixedVector<N> &lhs,
                           const Int8FixedVector<N> &rhs,
                           const FloatFixedVector<N> &wgt)
    {
        return mercury::internal::EuclideanDistance(lhs.data(), rhs.data(),
                                                    wgt.data(), N);
    }

    //! Compute the weighted squared euclidean distance between two vectors
    template <size_t N>
    static float SquaredEuclidean(const Int8FixedVector<N> &lhs,
                                  const Int8FixedVector<N> &rhs,
                                  const FloatFixedVector<N> &wgt)
    {
        return mercury::internal::SquaredEuclideanDistance(
            lhs.data(), rhs.data(), wgt.data(), N);
    }

    //! Compute the normalized euclidean distance between two vectors
    template <size_t N>
    static float NormalizedEuclidean(const Int8FixedVector<N> &lhs,
                                     const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::NormalizedEuclideanDistance(lhs.data(),
                                                              rhs.data(), N);
    }

    //! Compute the normalized squared euclidean distance between two vectors
    template <size_t N>
    static float NormalizedSquaredEuclidean(const Int8FixedVector<N> &lhs,
                                            const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::NormalizedSquaredEuclideanDistance(
            lhs.data(), rhs.data(), N);
    }

    //! Compute the manhattan distance between two vectors
    template <size_t N>
    static float Manhattan(const Int8FixedVector<N> &lhs,
                           const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::ManhattanDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the chebyshev distance between two vectors
    template <size_t N>
    static float Chebyshev(const Int8FixedVector<N> &lhs,
                           const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::ChebyshevDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the chessboard distance between two vectors
    template <size_t N>
    static float Chessboard(const Int8FixedVector<N> &lhs,
                            const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::ChessboardDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the cosine distance between two vectors
    template <size_t N>
    static float Cosine(const Int8FixedVector<N> &lhs,
                        const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::CosineDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the inner product between two vectors
    template <size_t N>
    static float InnerProduct(const Int8FixedVector<N> &lhs,
                              const Int8FixedVector<N> &rhs)
    {
        return -mercury::internal::InnerProduct(lhs.data(), rhs.data(), N);
    }

    //! Compute the canberra distance between two vectors
    template <size_t N>
    static float Canberra(const Int8FixedVector<N> &lhs,
                          const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::CanberraDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the bray-curtis distance between two vectors
    template <size_t N>
    static float BrayCurtis(const Int8FixedVector<N> &lhs,
                            const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::BrayCurtisDistance(lhs.data(), rhs.data(), N);
    }

    //! Compute the correlation distance between two vectors
    template <size_t N>
    static float Correlation(const Int8FixedVector<N> &lhs,
                             const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::CorrelationDistance(lhs.data(), rhs.data(),
                                                      N);
    }

    //! Compute the binary distance between two vectors
    template <size_t N>
    static float Binary(const Int8FixedVector<N> &lhs,
                        const Int8FixedVector<N> &rhs)
    {
        return mercury::internal::BinaryDistance(lhs.data(), rhs.data(), N);
    }
};

} // namespace mercury

#endif // __MERCURY_UTILITY_DISTANCE_H__
