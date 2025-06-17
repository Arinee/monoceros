#ifndef __INDEX_META_HELPER_H__
#define __INDEX_META_HELPER_H__

#include "src/core/framework/index_meta.h"
#include <string>
#include <iostream>

MERCURY_NAMESPACE_BEGIN(core);

class IndexMetaHelper
{
public:
    static std::string toString(mercury::core::IndexMeta::FeatureTypes type)
    {
        switch(type) {
            case mercury::core::IndexMeta::kTypeHalfFloat:
                return std::string("kTypeHalfFloat");
            case mercury::core::IndexMeta::kTypeFloat:
                return std::string("kTypeFloat");
            case mercury::core::IndexMeta::kTypeInt16:
                return std::string("kTypeInt16");
            case mercury::core::IndexMeta::kTypeInt8:
                return std::string("kTypeInt8");
            case mercury::core::IndexMeta::kTypeBinary:
                return std::string("kTypeBinary");
            default:
                return std::string("NotSupportedType");
        }
    }

    static std::string toString(mercury::core::IndexDistance::Methods method)
    {
        switch(method) {
            case mercury::core::IndexDistance::kMethodFloatSquaredEuclidean:
                return std::string("kMethodFloatSquaredEuclidean");
            case mercury::core::IndexDistance::kMethodHalfFloatSquaredEuclidean:
                return std::string("kMethodHalfFloatSquaredEuclidean");
            case mercury::core::IndexDistance::kMethodInt16SquaredEuclidean:
                return std::string("kMethodInt16SquaredEuclidean");
            case mercury::core::IndexDistance::kMethodInt8SquaredEuclidean:
                return std::string("kMethodInt8SquaredEuclidean");
            case mercury::core::IndexDistance::kMethodHalfFloatInnerProduct:
                return std::string("kMethodHalfFloatInnerProduct");
            case mercury::core::IndexDistance::kMethodFloatInnerProduct:
                return std::string("kMethodFloatInnerProduct");
            case mercury::core::IndexDistance::kMethodInt16InnerProduct:
                return std::string("kMethodInt16InnerProduct");
            case mercury::core::IndexDistance::kMethodInt8InnerProduct:
                return std::string("kMethodInt8InnerProduct");
            case mercury::core::IndexDistance::kMethodBinaryHamming:
                return std::string("kMethodBinaryHamming");
            default:
                return std::string("NotSupportedMethod");
        }
    }

    static std::string toString(mercury::core::IndexMeta meta)
    {
        char buffer[1024];
        snprintf(buffer, 1024, "IndexMeta: type[%s] method[%s] dimension[%lu]",
                toString(meta.type()).c_str(), toString(meta.method()).c_str(), meta.dimension());
        return std::string(buffer);
    }

    static bool parseFrom(const std::string &type,
                          const std::string &method,
                          const size_t dimension,
                          mercury::core::IndexMeta &meta)
    {
        auto featureType = mercury::core::IndexMeta::kTypeUnknown;
        if (type == std::string("half")) {
            featureType = mercury::core::IndexMeta::kTypeHalfFloat;
        } else if (type == std::string("float")) {
            featureType = mercury::core::IndexMeta::kTypeFloat;
        } else if (type == std::string("int16")) {
            featureType = mercury::core::IndexMeta::kTypeInt16;
        } else if (type == std::string("int8")) {
            featureType = mercury::core::IndexMeta::kTypeInt8;
        } else if (type == std::string("binary")) {
            featureType = mercury::core::IndexMeta::kTypeBinary;
        } else {
            std::cerr << "Not supported type: " << type << std::endl;
            return false;
        }

        meta.setMeta(featureType, dimension);
        if (method == std::string("L2")) {
            if (featureType == mercury::core::IndexMeta::kTypeFloat) {
                meta.setMethod(mercury::core::IndexDistance::kMethodFloatSquaredEuclidean);
            } else if (featureType == mercury::core::IndexMeta::kTypeHalfFloat) {
                meta.setMethod(mercury::core::IndexDistance::kMethodHalfFloatSquaredEuclidean);
            } else if (featureType == mercury::core::IndexMeta::kTypeInt16) {
                meta.setMethod(mercury::core::IndexDistance::kMethodInt16SquaredEuclidean);
            } else if (featureType == mercury::core::IndexMeta::kTypeInt8) {
                meta.setMethod(mercury::core::IndexDistance::kMethodInt8SquaredEuclidean);
            } else {
                std::cerr << "Not supported type(" << type << 
                    ") for L2" << std::endl;
                return false;
            }
        } else if (method == std::string("IP"))  {
            if (featureType == mercury::core::IndexMeta::kTypeHalfFloat) {
                meta.setMethod(mercury::core::IndexDistance::kMethodHalfFloatInnerProduct);
            } else if (featureType == mercury::core::IndexMeta::kTypeFloat) {
                meta.setMethod(mercury::core::IndexDistance::kMethodFloatInnerProduct);
            } else if (featureType == mercury::core::IndexMeta::kTypeInt16) {
                meta.setMethod(mercury::core::IndexDistance::kMethodInt16InnerProduct);
            } else if (featureType == mercury::core::IndexMeta::kTypeInt8) {
                meta.setMethod(mercury::core::IndexDistance::kMethodInt8InnerProduct);
            } else {
                std::cerr << "Not supported type(" << type 
                    << ") for IP" << std::endl;
                return false;
            }
        } else if (method == std::string("HAMMING"))  {
            if (featureType == mercury::core::IndexMeta::kTypeBinary) {
                meta.setMethod(mercury::core::IndexDistance::kMethodBinaryHamming);
            } else {
                std::cerr << "Not supported type(" << type 
                    << ") for hamming" << std::endl;
                return false;
            }
        } else {
            std::cerr << "Not supported method: " << method << std::endl;
            return false;
        }
        return true;
    }

};

MERCURY_NAMESPACE_END(core);
#endif //__INDEX_META_HELPER_H__
