/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     vector_variance.h
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Vector Variance
 */

#ifndef __MERCURY_ALGORITHM_VECTOR_VARIANCE_H__
#define __MERCURY_ALGORITHM_VECTOR_VARIANCE_H__

#include "../utility/vector.h"
#include <type_traits>
#include <vector>

namespace mercury {

/*! Vector Variance
 */
struct VectorVariance
{
    //! Destructor
    virtual ~VectorVariance(void) {}

    //! Reset accumulator
    virtual void reset(void) = 0;

    //! Plus a vector
    virtual bool plus(const void *vec, size_t len) = 0;

    //! Retrieve the mean of vectors
    virtual bool mean(void *out, size_t len) const = 0;

    //! Retrieve the mean of vectors
    virtual void mean(std::string *out) const = 0;

    //! Retrieve the variance of vectors
    virtual void variance(std::vector<double> *out) const = 0;

    //! Merge another vector variance
    virtual bool merge(const VectorVariance &rhs) = 0;
};

/*! Numerical Vector Mean
 */
template <typename T,
          typename =
              typename std::enable_if<std::is_arithmetic<T>::value>::type>
class NumericalVectorVariance : public VectorVariance
{
public:
    //! Constructor
    NumericalVectorVariance(void)
        : _dimension(0), _count(0), _accums(), _squ_accums()
    {
    }

    //! Constructor
    NumericalVectorVariance(size_t dim)
        : _dimension(dim), _count(0), _accums(), _squ_accums()
    {
        _accums.resize(dim, 0.0);
        _squ_accums.resize(dim, 0.0);
    }

    //! Retrieve count of accumulator
    size_t count(void) const
    {
        return _count;
    }

    //! Retrieve dimension of accumulator
    size_t dimension(void) const
    {
        return _dimension;
    }

    //! Reset accumulator
    void reset(size_t dim)
    {
        _count = 0u;
        _dimension = dim;
        _accums.clear();
        _accums.resize(_dimension, 0.0);
        _squ_accums.clear();
        _squ_accums.resize(_dimension, 0.0);
    }

    //! Reset accumulator
    virtual void reset(void)
    {
        _count = 0u;
        _accums.clear();
        _accums.resize(_dimension, 0.0);
        _squ_accums.clear();
        _squ_accums.resize(_dimension, 0.0);
    }

    //! Plus a vector
    virtual bool plus(const void *vec, size_t len)
    {
        if (_accums.size() * sizeof(T) != len) {
            return false;
        }
        for (size_t i = 0; i < _accums.size(); ++i) {
            T val = *(static_cast<const T *>(vec) + i);
            _accums[i] += val;
            _squ_accums[i] += (val * val);
        }
        ++_count;
        return true;
    }

    //! Retrieve the mean of vectors
    virtual bool mean(void *out, size_t len) const
    {
        if (_accums.size() * sizeof(T) != len) {
            return false;
        }
        for (size_t i = 0; i < _accums.size(); ++i) {
            *(static_cast<T *>(out) + i) = static_cast<T>(_accums[i] / _count);
        }
        return true;
    }

    //! Retrieve the mean of vectors
    virtual void mean(std::string *out) const
    {
        Vector<T> &vec = *static_cast<Vector<T> *>(out);
        vec.resize(_accums.size());
        for (size_t i = 0; i < _accums.size(); ++i) {
            vec[i] = static_cast<T>(_accums[i] / _count);
        }
    }

    //! Retrieve the variance of vectors
    virtual void variance(std::vector<double> *out) const
    {
        out->resize(_accums.size());
        for (size_t i = 0; i < _accums.size(); ++i) {
            double val = _accums[i] / _count;
            (*out)[i] = _squ_accums[i] / _count - val * val;
        }
    }

    //! Merge another vector variance
    virtual bool merge(const VectorVariance &rhs)
    {
        const NumericalVectorVariance<T> &src =
            dynamic_cast<const NumericalVectorVariance<T> &>(rhs);

        if (_dimension != src._dimension) {
            return false;
        }
        _count += src._count;
        for (size_t i = 0; i < _dimension; ++i) {
            _accums[i] += src._accums[i];
            _squ_accums[i] += src._squ_accums[i];
        }
        return true;
    }

private:
    //! Members
    size_t _dimension;
    size_t _count;
    std::vector<double> _accums;
    std::vector<double> _squ_accums;
};

/*! Binary Vector Variance
 */
class BinaryVectorVariance : public VectorVariance
{
public:
    //! Constructor
    BinaryVectorVariance(void) : _dimension(0), _count(0), _accums() {}

    //! Constructor
    BinaryVectorVariance(size_t dim) : _dimension(dim), _count(0), _accums()
    {
        _accums.resize(dim);
    }

    //! Retrieve count of accumulator
    size_t count(void) const
    {
        return _count;
    }

    //! Retrieve dimension of accumulator
    size_t dimension(void) const
    {
        return _dimension;
    }

    //! Reset accumulator
    void reset(size_t dim)
    {
        _count = 0u;
        _dimension = dim;
        _accums.clear();
        _accums.resize(_dimension);
    }

    //! Reset accumulator
    virtual void reset(void)
    {
        _count = 0u;
        _accums.clear();
        _accums.resize(_dimension);
    }

    //! Plus a vector
    virtual bool plus(const void *vec, size_t len)
    {
        if (_accums.size() != len * 8) {
            return false;
        }

        const uint8_t *bits = reinterpret_cast<const uint8_t *>(vec);
        for (size_t i = 0; i < _accums.size(); ++i) {
            if (bits[i >> 3] & static_cast<uint8_t>(1 << (i & 0x7))) {
                _accums[i] += 1;
            }
        }
        ++_count;
        return true;
    }

    //! Retrieve the mean of vectors
    virtual bool mean(void *out, size_t len) const
    {
        if (_accums.size() != len * 8) {
            return false;
        }
        memset(out, 0, len);

        uint8_t *bits = reinterpret_cast<uint8_t *>(out);
        size_t half_count = _count >> 1;
        for (size_t i = 0; i < _accums.size(); ++i) {
            if (_accums[i] > half_count) {
                bits[i >> 3] |= static_cast<uint8_t>(1 << (i & 0x7));
            }
        }
        return true;
    }

    //! Retrieve the mean of vectors
    virtual void mean(std::string *out) const
    {
        out->clear();
        out->resize((_accums.size() + 7) / 8);

        uint8_t *bits =
            reinterpret_cast<uint8_t *>(const_cast<char *>(out->data()));
        size_t half_count = _count >> 1;
        for (size_t i = 0; i < _accums.size(); ++i) {
            if (_accums[i] > half_count) {
                bits[i >> 3] |= static_cast<uint8_t>(1 << (i & 0x7));
            }
        }
    }

    //! Retrieve the variance of vectors
    virtual void variance(std::vector<double> *out) const
    {
        out->resize(_accums.size());

        for (size_t i = 0; i < _accums.size(); ++i) {
            double val = (double)_accums[i] / _count;
            (*out)[i] = val - val * val;
        }
    }

    //! Merge another vector variance
    virtual bool merge(const VectorVariance &rhs)
    {
        const BinaryVectorVariance &src =
            dynamic_cast<const BinaryVectorVariance &>(rhs);

        if (_dimension != src._dimension) {
            return false;
        }
        _count += src._count;
        for (size_t i = 0; i < _dimension; ++i) {
            _accums[i] += src._accums[i];
        }
        return true;
    }

private:
    //! Members
    size_t _dimension;
    size_t _count;
    std::vector<size_t> _accums;
};

} // namespace mercury

#endif // __MERCURY_ALGORITHM_VECTOR_VARIANCE_H__
