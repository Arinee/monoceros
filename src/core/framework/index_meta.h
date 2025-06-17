/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_meta.h
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Index Meta
 */

#ifndef __MERCURY_INDEX_META_H__
#define __MERCURY_INDEX_META_H__

#include "index_distance.h"
// #include "utility/json.h"
#include "src/core/common/common.h"
#include <string>

MERCURY_NAMESPACE_BEGIN(core);

struct VectorHolder;
struct VectorHolder;

/*! Index Meta
 */
class IndexMeta
{
public:
    /*! Feature Types
     */
    enum FeatureTypes
    {
        kTypeUnknown = 0,
        kTypeBinary = 1,
        kTypeFloat = 2,
        kTypeDouble = 3,
        kTypeInt8 = 4,
        kTypeInt16 = 5,
        kTypeHalfFloat = 6
    };

    //! Constructor
    IndexMeta(void)
        : _measure(), _ip_measure(), _method(IndexDistance::kMethodUnknown), _type(FeatureTypes::kTypeUnknown), _dimension(0),
          _element_size(0)
    {
    }

    //! Constructor
    IndexMeta(const IndexMeta &rhs)
        : _measure(rhs._measure), _ip_measure(rhs._ip_measure), _method(rhs._method), _type(rhs._type), _dimension(rhs._dimension),
          _element_size(rhs._element_size)
    //, _attachment(rhs._attachment)
    {
    }

    //! Constructor
    IndexMeta(IndexMeta &&rhs)
        : _measure(std::move(rhs._measure)), _ip_measure(std::move(rhs._ip_measure)), _method(rhs._method), _type(rhs._type), _dimension(rhs._dimension),
          _element_size(rhs._element_size) //, _attachment(rhs._attachment)
    {
    }

    //! Assignment
    IndexMeta &operator=(const IndexMeta &rhs)
    {
        _type = rhs._type;
        _measure = rhs._measure;
        _ip_measure = rhs._ip_measure;
        _method = rhs._method;
        _dimension = rhs._dimension;
        _element_size = rhs._element_size;
        //_attachment = rhs._attachment;
        return *this;
    }

    //! Assignment
    IndexMeta &operator=(IndexMeta &&rhs)
    {
        _type = rhs._type;
        _measure = std::move(rhs._measure);
        _ip_measure = std::move(rhs._ip_measure);
        _method = rhs._method;
        _dimension = rhs._dimension;
        _element_size = rhs._element_size;
        //_attachment = rhs._attachment;
        return *this;
    }

    //! Retrieve dimension
    size_t dimension(void) const
    {
        return _dimension;
    }

    //! Retrieve the distance between two features
    float distance(const void *lhs, const void *rhs) const
    {
        return _measure(lhs, rhs, _element_size);
    }

    //! Retrieve the float inner product between two features
    float fip(const void *lhs, const void *rhs) const
    {
        return _ip_measure(lhs, rhs, _element_size);
    }

    //! Retrieve the distance measure
    IndexDistance::Measure measure(void) const
    {
        return _measure;
    }

    //! Retrieve the distance method
    IndexDistance::Methods method(void) const
    {
        return _method;
    }

    //! Retrieve type information
    FeatureTypes type(void) const
    {
        return _type;
    }

    ////! Retrieve attachment
    // const JsonObject &attachment(void) const
    //{
    //     return _attachment;
    // }

    //! Retrieve element size in bytes
    size_t sizeofElement(bool enable_mips = false) const
    {
        if (enable_mips) {
            return IndexMeta::Sizeof(_type, _dimension + 1);
        }
        return _element_size;
    }

    //! Set dimension of feature
    void setDimension(size_t dim)
    {
        _dimension = dim;
        _element_size = IndexMeta::Sizeof(_type, _dimension);
    }

    //! Set type of feature
    void setType(FeatureTypes val)
    {
        _type = val;
        _element_size = IndexMeta::Sizeof(_type, _dimension);
    }

    //! Set meta information of feature
    void setMeta(FeatureTypes val, size_t dim)
    {
        _dimension = dim;
        _type = val;
        _element_size = IndexMeta::Sizeof(_type, _dimension);
    }

    //! Set the distance measure
    void setMethod(IndexDistance::Methods meth)
    {
        _method = meth;
        this->updateMeasure();
    }

    //! Set the distance measure
    template <typename T>
    void setMethod(const T &func)
    {
        _measure = func;
        _method = IndexDistance::kMethodCustom;
    }

    ////! Set attachment of meta
    // void setAttachment(const JsonObject &obj)
    //{
    //     _attachment = obj;
    // }

    //! Test if meta is valid
    bool isValid(void) const
    {
        FeatureTypes val = IndexMeta::Typeof(_method);
        if (val != FeatureTypes::kTypeUnknown && val != _type) {
            return false;
        }
        if (_element_size != IndexMeta::Sizeof(_type, _dimension)) {
            return false;
        }
        return true;
    }

    //! Test if matchs the holder
    bool isMatched(const VectorHolder &holder) const;

    //! Serialize meta information into buffer
    void serialize(std::string *out) const;

    //! Derialize meta information from buffer
    bool deserialize(const void *data, size_t len);

    float getMipsScoreOffline(const void *data, const void *centroid_data, const void *mips_norm) const
    {
        float score = 0.0;
        float data_norm_square = -_ip_measure(data, data, _element_size);
        switch (_type) {
        case FeatureTypes::kTypeHalfFloat: {
            float mips_norm_float = *(half_float::half *)mips_norm;
            float mips_norm_sqaure = mips_norm_float*mips_norm_float;
            float last_centroid = *((half_float::half *)centroid_data + _dimension);
            float last_norm = sqrt(mips_norm_sqaure - data_norm_square) - last_centroid;
            score = last_norm * last_norm;
        } break;
        case FeatureTypes::kTypeFloat: {
            float mips_norm_sqaure = -_ip_measure(mips_norm, mips_norm, 4);
            float last_centroid = *((float *)centroid_data + _dimension);
            float last_norm = sqrt(mips_norm_sqaure - data_norm_square) - last_centroid;
            score = last_norm * last_norm;
        } break;
        default:
            break;
        }

        return score;
    }

    float getMipsScoreOnline(void *centroid_data) const
    {
        float score = 0.0;
        switch (_type) {
        case FeatureTypes::kTypeHalfFloat: {
            float last_centroid = *((half_float::half *)centroid_data + _dimension);
            score = last_centroid * last_centroid;
        } break;
        case FeatureTypes::kTypeFloat: {
            float last_centroid = *((float *)centroid_data + _dimension);
            score = last_centroid * last_centroid;
        } break;
        default:
            break;
        }

        return score;
    }

    //! Calculate size of feature
    static size_t Sizeof(FeatureTypes val, size_t dim)
    {
        switch (val) {
        case FeatureTypes::kTypeUnknown:
            return 0;
        case FeatureTypes::kTypeBinary:
            return (dim / 8);
        case FeatureTypes::kTypeHalfFloat:
            return (dim * sizeof(half_float::half));
        case FeatureTypes::kTypeFloat:
            return (dim * sizeof(float));
        case FeatureTypes::kTypeDouble:
            return (dim * sizeof(double));
        case FeatureTypes::kTypeInt8:
            return (dim * sizeof(int8_t));
        case FeatureTypes::kTypeInt16:
            return (dim * sizeof(int16_t));
        }
        return 0;
    }

    //! Calculate type of feature
    static FeatureTypes Typeof(size_t dim, size_t size)
    {
        if (size % dim == 0) {
            switch (size / dim) {
            case 8:
                return FeatureTypes::kTypeDouble;
            case 4:
                return FeatureTypes::kTypeFloat;
            // case 2: can be half or int16
            //     return FeatureTypes::kTypeInt16;
            case 1:
                return FeatureTypes::kTypeInt8;
            }
        } else if (dim % size == 0) {
            if (dim / size == 8) {
                return FeatureTypes::kTypeBinary;
            }
        }
        return FeatureTypes::kTypeUnknown;
    }

    //! Calculate type of method
    static FeatureTypes Typeof(IndexDistance::Methods method);

protected:
    //! Update distance method
    void updateMeasure(void)
    {
        _measure = IndexDistance::EmbodyMeasure(_method);
        if (_type == FeatureTypes::kTypeHalfFloat) {
            _ip_measure = IndexDistance::EmbodyMeasure(IndexDistance::kMethodHalfFloatInnerProduct);
        }
        else {
            _ip_measure = IndexDistance::EmbodyMeasure(IndexDistance::kMethodFloatInnerProduct);
        }
    }

private:
    IndexDistance::Measure _measure;
    IndexDistance::Measure _ip_measure;
    IndexDistance::Methods _method;
    FeatureTypes _type;
    size_t _dimension;
    size_t _element_size;
    // JsonObject _attachment;
};

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_INDEX_META_H__
