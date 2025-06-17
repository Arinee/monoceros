/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_distance.h
 *   \author   Hechong.xyf
 *   \date     Jan 2018
 *   \version  1.0.0
 *   \brief    Interface of Index Distance
 */

#ifndef __MERCURY_INDEX_DISTANCE_H__
#define __MERCURY_INDEX_DISTANCE_H__

#include "utility/distance.h"
#include <functional>
#include "src/core/common/common.h"

MERCURY_NAMESPACE_BEGIN(core);
/*! Index Distance
 */
struct IndexDistance
{
    /*! Distance Measure
     */
    typedef std::function<float(const void *, const void *, size_t)> Measure;

    /*! Distance Methods
     */
    enum Methods
    {
        kMethodUnknown = 0,
        kMethodCustom = 1,

        kMethodFloatEuclidean = 31,
        kMethodFloatSquaredEuclidean = 32,
        kMethodFloatNormalizedEuclidean = 33,
        kMethodFloatNormalizedSquaredEuclidean = 34,
        kMethodFloatManhattan = 35,
        kMethodFloatChebyshev = 36,
        kMethodFloatChessboard = 37,
        kMethodFloatCosine = 38,
        kMethodFloatCanberra = 39,
        kMethodFloatBrayCurtis = 40,
        kMethodFloatCorrelation = 41,
        kMethodFloatBinary = 42,
        kMethodFloatInnerProduct = 43,

        // half
        kMethodHalfFloatInnerProduct = 44,
        kMethodHalfFloatSquaredEuclidean = 45,

        // kMethodDoubleEuclidean = 1031,
        // kMethodDoubleSquaredEuclidean = 1032,
        // kMethodDoubleNormalizedEuclidean = 1033,
        // kMethodDoubleNormalizedSquaredEuclidean = 1034,
        // kMethodDoubleManhattan = 1035,
        // kMethodDoubleChebyshev = 1036,
        // kMethodDoubleChessboard = 1037,
        // kMethodDoubleCosine = 1038,
        // kMethodDoubleCanberra = 1039,
        // kMethodDoubleBrayCurtis = 1040,
        // kMethodDoubleCorrelation = 1041,
        // kMethodDoubleBinary = 1042,
        // kMethodDoubleInnerProduct = 1043,

        kMethodInt8Euclidean = 2031,
        kMethodInt8SquaredEuclidean = 2032,
        kMethodInt8NormalizedEuclidean = 2033,
        kMethodInt8NormalizedSquaredEuclidean = 2034,
        kMethodInt8Manhattan = 2035,
        kMethodInt8Chebyshev = 2036,
        kMethodInt8Chessboard = 2037,
        kMethodInt8Cosine = 2038,
        kMethodInt8Canberra = 2039,
        kMethodInt8BrayCurtis = 2040,
        kMethodInt8Correlation = 2041,
        kMethodInt8Binary = 2042,
        kMethodInt8InnerProduct = 2043,

        kMethodInt16Euclidean = 3031,
        kMethodInt16SquaredEuclidean = 3032,
        kMethodInt16NormalizedEuclidean = 3033,
        kMethodInt16NormalizedSquaredEuclidean = 3034,
        kMethodInt16Manhattan = 3035,
        kMethodInt16Chebyshev = 3036,
        kMethodInt16Chessboard = 3037,
        kMethodInt16Cosine = 3038,
        kMethodInt16Canberra = 3039,
        kMethodInt16BrayCurtis = 3040,
        kMethodInt16Correlation = 3041,
        kMethodInt16Binary = 3042,
        kMethodInt16InnerProduct = 3043,

        kMethodBinaryHamming = 8011,
        kMethodBinaryJaccard = 8012,
        kMethodBinaryMatching = 8013,
        kMethodBinaryDice = 8014,
        kMethodBinaryRogersTanimoto = 8015,
        kMethodBinaryRussellRao = 8016,
        kMethodBinarySokalMichener = 8017,
        kMethodBinarySokalSneath = 8018,
        kMethodBinaryYule = 8019,
    };

    //! Distance normalizer
    class Normalizer
    {
    public:
        //! Constructor
        Normalizer(void) : _method(IndexDistance::kMethodUnknown) {}

        //! Constructor
        Normalizer(IndexDistance::Methods method) : _method(method) {}

        //! Destructor
        ~Normalizer(void) {}

        //! Function call
        float operator()(float score) const;

    private:
        //! Members
        IndexDistance::Methods _method;
    };

    //! Embody the measure
    static Measure EmbodyMeasure(Methods method);
};

/*! Hamming Distance Calculator
 */
struct BinaryHammingCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs) const
    {
        return Distance::Hamming<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::HammingDistance(
            reinterpret_cast<const uint32_t *>(lhs),
            reinterpret_cast<const uint32_t *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodBinaryHamming;
    }
};

/*! Jaccard Distance Calculator
 */
struct BinaryJaccardCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs) const
    {
        return Distance::Jaccard<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::JaccardDistance(
            reinterpret_cast<const uint32_t *>(lhs),
            reinterpret_cast<const uint32_t *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodBinaryJaccard;
    }
};

/*! Matching Distance Calculator
 */
struct BinaryMatchingCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs) const
    {
        return Distance::Matching<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::MatchingDistance(
            reinterpret_cast<const uint32_t *>(lhs),
            reinterpret_cast<const uint32_t *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodBinaryMatching;
    }
};

/*! Dice Distance Calculator
 */
struct BinaryDiceCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs) const
    {
        return Distance::Dice<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::DiceDistance(reinterpret_cast<const uint32_t *>(lhs),
                                      reinterpret_cast<const uint32_t *>(rhs),
                                      (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodBinaryDice;
    }
};

/*! Rogers-Tanimoto Distance Calculator
 */
struct BinaryRogersTanimotoCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs) const
    {
        return Distance::RogersTanimoto<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::RogersTanimotoDistance(
            reinterpret_cast<const uint32_t *>(lhs),
            reinterpret_cast<const uint32_t *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodBinaryRogersTanimoto;
    }
};

/*! Russell-Rao Distance Calculator
 */
struct BinaryRussellRaoCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs) const
    {
        return Distance::RussellRao<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::RussellRaoDistance(
            reinterpret_cast<const uint32_t *>(lhs),
            reinterpret_cast<const uint32_t *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodBinaryRussellRao;
    }
};

/*! Sokal-Michener Distance Calculator
 */
struct BinarySokalMichenerCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs) const
    {
        return Distance::SokalMichener<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::SokalMichenerDistance(
            reinterpret_cast<const uint32_t *>(lhs),
            reinterpret_cast<const uint32_t *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodBinarySokalMichener;
    }
};

/*! Sokal-Sneath Distance Calculator
 */
struct BinarySokalSneathCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs) const
    {
        return Distance::SokalSneath<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::SokalSneathDistance(
            reinterpret_cast<const uint32_t *>(lhs),
            reinterpret_cast<const uint32_t *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodBinarySokalSneath;
    }
};

/*! Yule Distance Calculator
 */
struct BinaryYuleCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FixedBitset<N> &lhs, const FixedBitset<N> &rhs) const
    {
        return Distance::Yule<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::YuleDistance(reinterpret_cast<const uint32_t *>(lhs),
                                      reinterpret_cast<const uint32_t *>(rhs),
                                      (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodBinaryYule;
    }
};

/*! Euclidean Distance Calculator
 */
struct FloatEuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::Euclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::EuclideanDistance(reinterpret_cast<const float *>(lhs),
                                           reinterpret_cast<const float *>(rhs),
                                           (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatEuclidean;
    }
};

/*! Squared Euclidean Distance Calculator
 */
struct FloatSquaredEuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::SquaredEuclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::SquaredEuclideanDistance(
            reinterpret_cast<const float *>(lhs),
            reinterpret_cast<const float *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatSquaredEuclidean;
    }
};

/*! Squared Euclidean Distance Calculator
 */
struct HalfFloatSquaredEuclideanCalculator
{
    //! Function call
    // template <size_t N>
    // float operator()(const FloatFixedVector<N> &lhs,
    //                  const FloatFixedVector<N> &rhs) const
    // {
    //     return Distance::SquaredEuclidean<N>(lhs, rhs);
    // }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::SquaredEuclideanDistance(
            reinterpret_cast<const half_float::half *>(lhs),
            reinterpret_cast<const half_float::half *>(rhs), (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodHalfFloatSquaredEuclidean;
    }
};

/*! Normalized Euclidean Distance Calculator
 */
struct FloatNormalizedEuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::NormalizedEuclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::NormalizedEuclideanDistance(
            reinterpret_cast<const float *>(lhs),
            reinterpret_cast<const float *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatNormalizedEuclidean;
    }
};

/*! Normalized Squared Euclidean Distance Calculator
 */
struct FloatNormalizedSquaredEuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::NormalizedSquaredEuclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::NormalizedSquaredEuclideanDistance(
            reinterpret_cast<const float *>(lhs),
            reinterpret_cast<const float *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatNormalizedSquaredEuclidean;
    }
};

/*! Manhattan Distance Calculator
 */
struct FloatManhattanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::Manhattan<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::ManhattanDistance(reinterpret_cast<const float *>(lhs),
                                           reinterpret_cast<const float *>(rhs),
                                           (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatManhattan;
    }
};

/*! Chebyshev Distance Calculator
 */
struct FloatChebyshevCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::Chebyshev<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::ChebyshevDistance(reinterpret_cast<const float *>(lhs),
                                           reinterpret_cast<const float *>(rhs),
                                           (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatChebyshev;
    }
};

/*! Chessboard Distance Calculator
 */
struct FloatChessboardCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::Chessboard<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::ChebyshevDistance(reinterpret_cast<const float *>(lhs),
                                           reinterpret_cast<const float *>(rhs),
                                           (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatChessboard;
    }
};


/*! Cosine Distance Calculator
 */
struct FloatCosineCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::Cosine<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::CosineDistance(reinterpret_cast<const float *>(lhs),
                                        reinterpret_cast<const float *>(rhs),
                                        (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatCosine;
    }
};

/*! Inner Product Distance Calculator
 */
struct FloatInnerProductCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::InnerProduct<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return -internal::InnerProduct(reinterpret_cast<const float *>(lhs),
                                       reinterpret_cast<const float *>(rhs),
                                       (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatInnerProduct;
    }
};

/*! Inner Half Product Distance Calculator
 */
struct HalfFloatInnerProductCalculator
{
    //! Function call
    // template <size_t N>
    // float operator()(const FloatFixedVector<N> &lhs,
    //                  const FloatFixedVector<N> &rhs) const
    // {
    //     return Distance::InnerProduct<N>(lhs, rhs);
    // }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return -internal::InnerProduct(reinterpret_cast<const half_float::half *>(lhs),
                                       reinterpret_cast<const half_float::half *>(rhs),
                                       (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodHalfFloatInnerProduct;
    }
};

/*! Canberra Distance Calculator
 */
struct FloatCanberraCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::Canberra<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::CanberraDistance(reinterpret_cast<const float *>(lhs),
                                          reinterpret_cast<const float *>(rhs),
                                          (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatCanberra;
    }
};

/*! BrayCurtis Distance Calculator
 */
struct FloatBrayCurtisCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::BrayCurtis<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::BrayCurtisDistance(
            reinterpret_cast<const float *>(lhs),
            reinterpret_cast<const float *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatBrayCurtis;
    }
};

/*! Correlation Distance Calculator
 */
struct FloatCorrelationCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::Correlation<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::CorrelationDistance(
            reinterpret_cast<const float *>(lhs),
            reinterpret_cast<const float *>(rhs), (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatCorrelation;
    }
};

/*! Binary Distance Calculator
 */
struct FloatBinaryCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const FloatFixedVector<N> &lhs,
                     const FloatFixedVector<N> &rhs) const
    {
        return Distance::Correlation<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::BinaryDistance(reinterpret_cast<const float *>(lhs),
                                        reinterpret_cast<const float *>(rhs),
                                        (size >> 2));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodFloatBinary;
    }
};

/*! Euclidean Distance Calculator
 */
struct Int16EuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::Euclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::EuclideanDistance(
            reinterpret_cast<const int16_t *>(lhs),
            reinterpret_cast<const int16_t *>(rhs), (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16Euclidean;
    }
};

/*! Squared Euclidean Distance Calculator
 */
struct Int16SquaredEuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::SquaredEuclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::SquaredEuclideanDistance(
            reinterpret_cast<const int16_t *>(lhs),
            reinterpret_cast<const int16_t *>(rhs), (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16SquaredEuclidean;
    }
};

/*! Normalized Euclidean Distance Calculator
 */
struct Int16NormalizedEuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::NormalizedEuclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::NormalizedEuclideanDistance(
            reinterpret_cast<const int16_t *>(lhs),
            reinterpret_cast<const int16_t *>(rhs), (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16NormalizedEuclidean;
    }
};

/*! Normalized Squared Euclidean Distance Calculator
 */
struct Int16NormalizedSquaredEuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::NormalizedSquaredEuclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::NormalizedSquaredEuclideanDistance(
            reinterpret_cast<const int16_t *>(lhs),
            reinterpret_cast<const int16_t *>(rhs), (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16NormalizedSquaredEuclidean;
    }
};

/*! Manhattan Distance Calculator
 */
struct Int16ManhattanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::Manhattan<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::ManhattanDistance(
            reinterpret_cast<const int16_t *>(lhs),
            reinterpret_cast<const int16_t *>(rhs), (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16Manhattan;
    }
};

/*! Chebyshev Distance Calculator
 */
struct Int16ChebyshevCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::Chebyshev<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::ChebyshevDistance(
            reinterpret_cast<const int16_t *>(lhs),
            reinterpret_cast<const int16_t *>(rhs), (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16Chebyshev;
    }
};

/*! Chessboard Distance Calculator
 */
struct Int16ChessboardCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::Chessboard<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::ChebyshevDistance(
            reinterpret_cast<const int16_t *>(lhs),
            reinterpret_cast<const int16_t *>(rhs), (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16Chessboard;
    }
};

/*! Cosine Distance Calculator
 */
struct Int16CosineCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::Cosine<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::CosineDistance(reinterpret_cast<const int16_t *>(lhs),
                                        reinterpret_cast<const int16_t *>(rhs),
                                        (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16Cosine;
    }
};

/*! Inner Product Distance Calculator
 */
struct Int16InnerProductCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::InnerProduct<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return -internal::InnerProduct(reinterpret_cast<const int16_t *>(lhs),
                                       reinterpret_cast<const int16_t *>(rhs),
                                       (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16InnerProduct;
    }
};

/*! Canberra Distance Calculator
 */
struct Int16CanberraCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::Canberra<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::CanberraDistance(
            reinterpret_cast<const int16_t *>(lhs),
            reinterpret_cast<const int16_t *>(rhs), (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16Canberra;
    }
};

/*! BrayCurtis Distance Calculator
 */
struct Int16BrayCurtisCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::BrayCurtis<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::BrayCurtisDistance(
            reinterpret_cast<const int16_t *>(lhs),
            reinterpret_cast<const int16_t *>(rhs), (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16BrayCurtis;
    }
};

/*! Correlation Distance Calculator
 */
struct Int16CorrelationCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::Correlation<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::CorrelationDistance(
            reinterpret_cast<const int16_t *>(lhs),
            reinterpret_cast<const int16_t *>(rhs), (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16Correlation;
    }
};

/*! Binary Distance Calculator
 */
struct Int16BinaryCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int16FixedVector<N> &lhs,
                     const Int16FixedVector<N> &rhs) const
    {
        return Distance::Correlation<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::BinaryDistance(reinterpret_cast<const int16_t *>(lhs),
                                        reinterpret_cast<const int16_t *>(rhs),
                                        (size >> 1));
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt16Binary;
    }
};

/*! Euclidean Distance Calculator
 */
struct Int8EuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::Euclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::EuclideanDistance(
            reinterpret_cast<const int8_t *>(lhs),
            reinterpret_cast<const int8_t *>(rhs), size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8Euclidean;
    }
};

/*! Squared Euclidean Distance Calculator
 */
struct Int8SquaredEuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::SquaredEuclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::SquaredEuclideanDistance(
            reinterpret_cast<const int8_t *>(lhs),
            reinterpret_cast<const int8_t *>(rhs), size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8SquaredEuclidean;
    }
};

/*! Normalized Euclidean Distance Calculator
 */
struct Int8NormalizedEuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::NormalizedEuclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::NormalizedEuclideanDistance(
            reinterpret_cast<const int8_t *>(lhs),
            reinterpret_cast<const int8_t *>(rhs), size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8NormalizedEuclidean;
    }
};

/*! Normalized Squared Euclidean Distance Calculator
 */
struct Int8NormalizedSquaredEuclideanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::NormalizedSquaredEuclidean<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::NormalizedSquaredEuclideanDistance(
            reinterpret_cast<const int8_t *>(lhs),
            reinterpret_cast<const int8_t *>(rhs), size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8NormalizedSquaredEuclidean;
    }
};

/*! Manhattan Distance Calculator
 */
struct Int8ManhattanCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::Manhattan<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::ManhattanDistance(
            reinterpret_cast<const int8_t *>(lhs),
            reinterpret_cast<const int8_t *>(rhs), size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8Manhattan;
    }
};

/*! Chebyshev Distance Calculator
 */
struct Int8ChebyshevCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::Chebyshev<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::ChebyshevDistance(
            reinterpret_cast<const int8_t *>(lhs),
            reinterpret_cast<const int8_t *>(rhs), size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8Chebyshev;
    }
};

/*! Chessboard Distance Calculator
 */
struct Int8ChessboardCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::Chessboard<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::ChebyshevDistance(
            reinterpret_cast<const int8_t *>(lhs),
            reinterpret_cast<const int8_t *>(rhs), size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8Chessboard;
    }
};


/*! Cosine Distance Calculator
 */
struct Int8CosineCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::Cosine<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::CosineDistance(reinterpret_cast<const int8_t *>(lhs),
                                        reinterpret_cast<const int8_t *>(rhs),
                                        size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8Cosine;
    }
};

/*! Inner Product Distance Calculator
 */
struct Int8InnerProductCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::InnerProduct<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return -internal::InnerProduct(reinterpret_cast<const int8_t *>(lhs),
                                       reinterpret_cast<const int8_t *>(rhs),
                                       size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8InnerProduct;
    }
};

/*! Canberra Distance Calculator
 */
struct Int8CanberraCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::Canberra<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::CanberraDistance(reinterpret_cast<const int8_t *>(lhs),
                                          reinterpret_cast<const int8_t *>(rhs),
                                          size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8Canberra;
    }
};

/*! BrayCurtis Distance Calculator
 */
struct Int8BrayCurtisCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::BrayCurtis<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::BrayCurtisDistance(
            reinterpret_cast<const int8_t *>(lhs),
            reinterpret_cast<const int8_t *>(rhs), size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8BrayCurtis;
    }
};

/*! Correlation Distance Calculator
 */
struct Int8CorrelationCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::Correlation<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::CorrelationDistance(
            reinterpret_cast<const int8_t *>(lhs),
            reinterpret_cast<const int8_t *>(rhs), size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8Correlation;
    }
};

/*! Binary Distance Calculator
 */
struct Int8BinaryCalculator
{
    //! Function call
    template <size_t N>
    float operator()(const Int8FixedVector<N> &lhs,
                     const Int8FixedVector<N> &rhs) const
    {
        return Distance::Correlation<N>(lhs, rhs);
    }

    //! Function call
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return internal::BinaryDistance(reinterpret_cast<const int8_t *>(lhs),
                                        reinterpret_cast<const int8_t *>(rhs),
                                        size);
    }

    //! Retrieve distance method
    IndexDistance::Methods method(void) const
    {
        return IndexDistance::kMethodInt8Binary;
    }
};

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_INDEX_DISTANCE_H__
