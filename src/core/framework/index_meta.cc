/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_meta.cc
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Index Meta
 */

#include "index_meta.h"
#include "vector_holder.h"
#include "vector_holder.h"

//debug
#include <iostream>
using namespace std;

MERCURY_NAMESPACE_BEGIN(core);

/*! Index Meta Buffer Format
 */
struct IndexMetaBufferFormat
{
    uint32_t format_size;
    uint16_t type;
    uint16_t method;
    uint32_t dimension;
    uint32_t _reserved[5];
    //uint32_t attachment_size;
    //uint32_t _reserved[4];
};

void IndexMeta::serialize(std::string *out) const
{
    IndexMetaBufferFormat format;
    format.format_size = sizeof(format);
    format.type = static_cast<uint16_t>(_type);
    format.method = static_cast<uint16_t>(_method);
    format.dimension = static_cast<uint32_t>(_dimension);
    //format.attachment_size = 0;
    memset(format._reserved, 0, sizeof(format._reserved));

    //if (!_attachment.isValid() || _attachment.empty()) {
        out->append(reinterpret_cast<const char *>(&format), sizeof(format));
    //} else {
    //    JsonString result = JsonValue(_attachment).asJsonString();
    //    format.attachment_size = static_cast<uint32_t>(result.size());
    //    out->append(reinterpret_cast<const char *>(&format), sizeof(format));
    //    out->append(result.data(), format.attachment_size);
    //}
}

bool IndexMeta::deserialize(const void *data, size_t len)
{
    const IndexMetaBufferFormat *format =
        reinterpret_cast<const IndexMetaBufferFormat *>(data);

    if (sizeof(IndexMetaBufferFormat) > len) {
        return false;
    }
    if (sizeof(IndexMetaBufferFormat) > format->format_size) {
        return false;
    }
    //if (format->format_size + format->attachment_size > len) {
    //    return false;
    //}

    _type = static_cast<IndexMeta::FeatureTypes>(format->type);
    _method = static_cast<IndexDistance::Methods>(format->method);
    _dimension = format->dimension;
    _element_size = IndexMeta::Sizeof(_type, _dimension);
    //_attachment = JsonObject();
    this->updateMeasure();

    // Read attachment
    //if (format->attachment_size) {
    //    std::string str(reinterpret_cast<const char *>(data) +
    //                        format->format_size,
    //                    format->attachment_size);
    //    JsonValue val;
    //    if (!val.parse(str)) {
    //        return false;
    //    }
    //    if (!val.isObject()) {
    //        return false;
    //    }
    //    _attachment = val.asObject();
    //}
    return true;
}

bool IndexMeta::isMatched(const VectorHolder &holder) const
{
    return (this->isValid() && holder.type() == _type &&
            holder.dimension() == _dimension &&
            holder.sizeofElement() == _element_size);
}

IndexMeta::FeatureTypes IndexMeta::Typeof(IndexDistance::Methods method)
{
    switch (method) {
    case IndexDistance::kMethodUnknown:
    case IndexDistance::kMethodCustom:
        break;

    case IndexDistance::kMethodBinaryHamming:
    case IndexDistance::kMethodBinaryJaccard:
    case IndexDistance::kMethodBinaryMatching:
    case IndexDistance::kMethodBinaryDice:
    case IndexDistance::kMethodBinaryRogersTanimoto:
    case IndexDistance::kMethodBinaryRussellRao:
    case IndexDistance::kMethodBinarySokalMichener:
    case IndexDistance::kMethodBinarySokalSneath:
    case IndexDistance::kMethodBinaryYule:
        return FeatureTypes::kTypeBinary;

    case IndexDistance::kMethodFloatEuclidean:
    case IndexDistance::kMethodFloatSquaredEuclidean:
    case IndexDistance::kMethodFloatNormalizedEuclidean:
    case IndexDistance::kMethodFloatNormalizedSquaredEuclidean:
    case IndexDistance::kMethodFloatManhattan:
    case IndexDistance::kMethodFloatChebyshev:
    case IndexDistance::kMethodFloatChessboard:
    case IndexDistance::kMethodFloatCosine:
    case IndexDistance::kMethodFloatInnerProduct:
    case IndexDistance::kMethodFloatCanberra:
    case IndexDistance::kMethodFloatBrayCurtis:
    case IndexDistance::kMethodFloatCorrelation:
    case IndexDistance::kMethodFloatBinary:
        return FeatureTypes::kTypeFloat;

    case IndexDistance::kMethodHalfFloatInnerProduct:
    case IndexDistance::kMethodHalfFloatSquaredEuclidean:
        return FeatureTypes::kTypeHalfFloat;
        
    case IndexDistance::kMethodInt16Euclidean:
    case IndexDistance::kMethodInt16SquaredEuclidean:
    case IndexDistance::kMethodInt16NormalizedEuclidean:
    case IndexDistance::kMethodInt16NormalizedSquaredEuclidean:
    case IndexDistance::kMethodInt16Manhattan:
    case IndexDistance::kMethodInt16Chebyshev:
    case IndexDistance::kMethodInt16Chessboard:
    case IndexDistance::kMethodInt16Cosine:
    case IndexDistance::kMethodInt16InnerProduct:
    case IndexDistance::kMethodInt16Canberra:
    case IndexDistance::kMethodInt16BrayCurtis:
    case IndexDistance::kMethodInt16Correlation:
    case IndexDistance::kMethodInt16Binary:
        return FeatureTypes::kTypeInt16;

    case IndexDistance::kMethodInt8Euclidean:
    case IndexDistance::kMethodInt8SquaredEuclidean:
    case IndexDistance::kMethodInt8NormalizedEuclidean:
    case IndexDistance::kMethodInt8NormalizedSquaredEuclidean:
    case IndexDistance::kMethodInt8Manhattan:
    case IndexDistance::kMethodInt8Chebyshev:
    case IndexDistance::kMethodInt8Chessboard:
    case IndexDistance::kMethodInt8Cosine:
    case IndexDistance::kMethodInt8InnerProduct:
    case IndexDistance::kMethodInt8Canberra:
    case IndexDistance::kMethodInt8BrayCurtis:
    case IndexDistance::kMethodInt8Correlation:
    case IndexDistance::kMethodInt8Binary:
        return FeatureTypes::kTypeInt8;
    }
    return FeatureTypes::kTypeUnknown;
}

MERCURY_NAMESPACE_END(core);
