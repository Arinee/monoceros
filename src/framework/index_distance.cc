/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_distance.cc
 *   \author   Hechong.xyf
 *   \date     Apr 2018
 *   \version  1.0.0
 *   \brief    Implementation of Index Distance
 */

#include "index_distance.h"

namespace mercury {

IndexDistance::Measure
IndexDistance::EmbodyMeasure(IndexDistance::Methods method)
{
    IndexDistance::Measure func;

    switch (method) {
    case IndexDistance::kMethodUnknown:
    case IndexDistance::kMethodCustom:
        break;
    case IndexDistance::kMethodBinaryHamming:
        func = BinaryHammingCalculator();
        break;
    case IndexDistance::kMethodBinaryJaccard:
        func = BinaryJaccardCalculator();
        break;
    case IndexDistance::kMethodBinaryMatching:
        func = BinaryMatchingCalculator();
        break;
    case IndexDistance::kMethodBinaryDice:
        func = BinaryDiceCalculator();
        break;
    case IndexDistance::kMethodBinaryRogersTanimoto:
        func = BinaryRogersTanimotoCalculator();
        break;
    case IndexDistance::kMethodBinaryRussellRao:
        func = BinaryRussellRaoCalculator();
        break;
    case IndexDistance::kMethodBinarySokalMichener:
        func = BinarySokalMichenerCalculator();
        break;
    case IndexDistance::kMethodBinarySokalSneath:
        func = BinarySokalSneathCalculator();
        break;
    case IndexDistance::kMethodBinaryYule:
        func = BinaryYuleCalculator();
        break;

    case IndexDistance::kMethodFloatEuclidean:
        func = FloatEuclideanCalculator();
        break;
    case IndexDistance::kMethodFloatSquaredEuclidean:
        func = FloatSquaredEuclideanCalculator();
        break;
    case IndexDistance::kMethodFloatNormalizedEuclidean:
        func = FloatNormalizedEuclideanCalculator();
        break;
    case IndexDistance::kMethodFloatNormalizedSquaredEuclidean:
        func = FloatNormalizedSquaredEuclideanCalculator();
        break;
    case IndexDistance::kMethodFloatManhattan:
        func = FloatManhattanCalculator();
        break;
    case IndexDistance::kMethodFloatChebyshev:
        func = FloatChebyshevCalculator();
        break;
    case IndexDistance::kMethodFloatChessboard:
        func = FloatChessboardCalculator();
        break;
    case IndexDistance::kMethodFloatCosine:
        func = FloatCosineCalculator();
        break;
    case IndexDistance::kMethodFloatInnerProduct:
        func = FloatInnerProductCalculator();
        break;
    case IndexDistance::kMethodFloatCanberra:
        func = FloatCanberraCalculator();
        break;
    case IndexDistance::kMethodFloatBrayCurtis:
        func = FloatBrayCurtisCalculator();
        break;
    case IndexDistance::kMethodFloatCorrelation:
        func = FloatCorrelationCalculator();
        break;
    case IndexDistance::kMethodFloatBinary:
        func = FloatBinaryCalculator();
        break;

    case IndexDistance::kMethodInt16Euclidean:
        func = Int16EuclideanCalculator();
        break;
    case IndexDistance::kMethodInt16SquaredEuclidean:
        func = Int16SquaredEuclideanCalculator();
        break;
    case IndexDistance::kMethodInt16NormalizedEuclidean:
        func = Int16NormalizedEuclideanCalculator();
        break;
    case IndexDistance::kMethodInt16NormalizedSquaredEuclidean:
        func = Int16NormalizedSquaredEuclideanCalculator();
        break;
    case IndexDistance::kMethodInt16Manhattan:
        func = Int16ManhattanCalculator();
        break;
    case IndexDistance::kMethodInt16Chebyshev:
        func = Int16ChebyshevCalculator();
        break;
    case IndexDistance::kMethodInt16Chessboard:
        func = Int16ChessboardCalculator();
        break;
    case IndexDistance::kMethodInt16Cosine:
        func = Int16CosineCalculator();
        break;
    case IndexDistance::kMethodInt16InnerProduct:
        func = Int16InnerProductCalculator();
        break;
    case IndexDistance::kMethodInt16Canberra:
        func = Int16CanberraCalculator();
        break;
    case IndexDistance::kMethodInt16BrayCurtis:
        func = Int16BrayCurtisCalculator();
        break;
    case IndexDistance::kMethodInt16Correlation:
        func = Int16CorrelationCalculator();
        break;
    case IndexDistance::kMethodInt16Binary:
        func = Int16BinaryCalculator();
        break;

    case IndexDistance::kMethodInt8Euclidean:
        func = Int8EuclideanCalculator();
        break;
    case IndexDistance::kMethodInt8SquaredEuclidean:
        func = Int8SquaredEuclideanCalculator();
        break;
    case IndexDistance::kMethodInt8NormalizedEuclidean:
        func = Int8NormalizedEuclideanCalculator();
        break;
    case IndexDistance::kMethodInt8NormalizedSquaredEuclidean:
        func = Int8NormalizedSquaredEuclideanCalculator();
        break;
    case IndexDistance::kMethodInt8Manhattan:
        func = Int8ManhattanCalculator();
        break;
    case IndexDistance::kMethodInt8Chebyshev:
        func = Int8ChebyshevCalculator();
        break;
    case IndexDistance::kMethodInt8Chessboard:
        func = Int8ChessboardCalculator();
        break;
    case IndexDistance::kMethodInt8Cosine:
        func = Int8CosineCalculator();
        break;
    case IndexDistance::kMethodInt8InnerProduct:
        func = Int8InnerProductCalculator();
        break;
    case IndexDistance::kMethodInt8Canberra:
        func = Int8CanberraCalculator();
        break;
    case IndexDistance::kMethodInt8BrayCurtis:
        func = Int8BrayCurtisCalculator();
        break;
    case IndexDistance::kMethodInt8Correlation:
        func = Int8CorrelationCalculator();
        break;
    case IndexDistance::kMethodInt8Binary:
        func = Int8BinaryCalculator();
        break;
    }
    return func;
}

float IndexDistance::Normalizer::operator()(float score) const
{
    switch (_method) {
    case IndexDistance::kMethodFloatInnerProduct:
    // case IndexDistance::kMethodDoubleInnerProduct:
    case IndexDistance::kMethodInt16InnerProduct:
    case IndexDistance::kMethodInt8InnerProduct:
        return (-score);

    case IndexDistance::kMethodFloatCosine:
    // case IndexDistance::kMethodDoubleCosine:
    case IndexDistance::kMethodInt16Cosine:
    case IndexDistance::kMethodInt8Cosine:
    case IndexDistance::kMethodFloatCorrelation:
    // case IndexDistance::kMethodDoubleCorrelation:
    case IndexDistance::kMethodInt16Correlation:
    case IndexDistance::kMethodInt8Correlation:
        return (1.0f - score);

    default:
        return score;
    }
}

} // namespace mercury
