/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     vector_mean.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Interface of mercury Vector Mean
 */

#ifndef __MERCURY_CLUSTER_VECTOR_MEAN_H__
#define __MERCURY_CLUSTER_VECTOR_MEAN_H__

#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

namespace mercury {

/*! Vector Mean
 */
struct VectorMean
{
    //! Destructor
    virtual ~VectorMean(void) {}

    //! Reset accumulator
    virtual void reset(void) = 0;

    //! Plus a vector
    virtual bool plus(const void *vec, size_t len) = 0;

    //! Retrieve the mean of vectors
    virtual bool mean(void *out, size_t len) const = 0;
};

/*! Numerical Vector Mean
 */
template <typename T,
          typename =
              typename std::enable_if<std::is_arithmetic<T>::value>::type>
class NumericalVectorMean : public VectorMean
{
public:
    //! Constructor
    NumericalVectorMean(void) : _dimension(0), _count(0), _accums() {}

    //! Constructor
    NumericalVectorMean(size_t dim) : _dimension(dim), _count(0), _accums()
    {
        _accums.resize(dim, 0.0);
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
    }

    //! Reset accumulator
    virtual void reset(void)
    {
        _count = 0u;
        _accums.clear();
        _accums.resize(_dimension, 0.0);
    }

    //! Plus a vector
    virtual bool plus(const void *vec, size_t len)
    {
        if (_accums.size() * sizeof(T) != len) {
            return false;
        }
        for (size_t i = 0; i < _accums.size(); ++i) {
            _accums[i] += *(static_cast<const T *>(vec) + i);
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
        if (_count == 0) return true;
        for (size_t i = 0; i < _accums.size(); ++i) {
            *(static_cast<T *>(out) + i) = static_cast<T>(_accums[i] / _count);
        }
        return true;
    }

private:
    //! Members
    size_t _dimension;
    size_t _count;
    std::vector<double> _accums;
};

/*! Numerical Vector Harmonic Mean
 */
template <typename T,
          typename =
              typename std::enable_if<std::is_arithmetic<T>::value>::type>
class NumericalVectorHarmonicMean : public VectorMean
{
public:
    //! Constructor
    NumericalVectorHarmonicMean(void) : _dimension(0), _count(0), _accums() {}

    //! Constructor
    NumericalVectorHarmonicMean(size_t dim)
        : _dimension(dim), _count(0), _accums()
    {
        _accums.resize(dim, 0.0);
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
    }

    //! Reset accumulator
    virtual void reset(void)
    {
        _count = 0u;
        _accums.clear();
        _accums.resize(_dimension, 0.0);
    }

    //! Plus a vector (harmonic)
    virtual bool plus(const void *vec, size_t len)
    {
        if (_accums.size() * sizeof(T) != len) {
            return false;
        }
        for (size_t i = 0; i < _accums.size(); ++i) {
            _accums[i] += 1.0 / *(static_cast<const T *>(vec) + i);
        }
        ++_count;
        return true;
    }

    //! Retrieve the mean of vectors (harmonic)
    virtual bool mean(void *out, size_t len) const
    {
        if (_accums.size() * sizeof(T) != len) {
            return false;
        }
        for (size_t i = 0; i < _accums.size(); ++i) {
            *(static_cast<T *>(out) + i) = static_cast<T>(_count / _accums[i]);
        }
        return true;
    }

private:
    //! Members
    size_t _dimension;
    size_t _count;
    std::vector<double> _accums;
};

/*! Numerical Vector Geometric Mean
 */
template <typename T,
          typename =
              typename std::enable_if<std::is_arithmetic<T>::value>::type>
class NumericalVectorGeometricMean : public VectorMean
{
public:
    //! Constructor
    NumericalVectorGeometricMean(void) : _dimension(0), _count(0), _accums() {}

    //! Constructor
    NumericalVectorGeometricMean(size_t dim)
        : _dimension(dim), _count(0), _accums()
    {
        _accums.resize(dim, 0.0);
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
    }

    //! Reset accumulator
    virtual void reset(void)
    {
        _count = 0u;
        _accums.clear();
        _accums.resize(_dimension, 0.0);
    }

    //! Plus a vector (geometric)
    virtual bool plus(const void *vec, size_t len)
    {
        if (_accums.size() * sizeof(T) != len) {
            return false;
        }
        for (size_t i = 0; i < _accums.size(); ++i) {
            _accums[i] += *(static_cast<const T *>(vec) + i);
        }
        ++_count;
        return true;
    }

    //! Retrieve the mean of vectors (geometric)
    virtual bool mean(void *out, size_t len) const
    {
        if (_accums.size() * sizeof(T) != len) {
            return false;
        }
        if (_count == 0) return true;
        for (size_t i = 0; i < _accums.size(); ++i) {
            *(static_cast<T *>(out) + i) =
                static_cast<T>(std::pow(_accums[i], 1.0 / _count));
        }
        return true;
    }

private:
    //! Members
    size_t _dimension;
    size_t _count;
    std::vector<double> _accums;
};

/*! Binary Vector Mean
 */
class BinaryVectorMean : public VectorMean
{
public:
    //! Constructor
    BinaryVectorMean(void) : _dimension(0), _count(0), _accums() {}

    //! Constructor
    BinaryVectorMean(size_t dim) : _dimension(dim), _count(0), _accums()
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

private:
    //! Members
    size_t _dimension;
    size_t _count;
    std::vector<size_t> _accums;
};

} // namespace mercury

#endif // __MERCURY_CLUSTER_VECTOR_MEAN_H__
