/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     fixed_vector.h
 *   \author   Hechong.xyf
 *   \date     Jan 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Fixed Vector
 */

#ifndef __MERCURY_UTILITY_FIXED_VECTOR_H__
#define __MERCURY_UTILITY_FIXED_VECTOR_H__

#include <cstddef>
#include <cstdint>

namespace mercury {

/*! Fixed Vector
 */
template <typename T, size_t N>
class FixedVector
{
public:
    enum
    {
        MAX_SIZE = N
    };

    //! Constructor
    template <typename... U>
    FixedVector(U... vals) : _data{ vals... }
    {
    }

    //! Overloaded operator []
    T &operator[](size_t i)
    {
        return _data[i];
    }

    //! Overloaded operator []
    constexpr const T &operator[](size_t i) const
    {
        return _data[i];
    }

    //! Retrieve data pointer
    T *data(void)
    {
        return _data;
    }

    //! Retrieve data pointer
    const T *data(void) const
    {
        return _data;
    }

    //! Retrieve count of elements in vector
    constexpr size_t size(void) const
    {
        return MAX_SIZE;
    }

    //! Convert a array pointer to vector pointer
    static FixedVector *Cast(T arr[N])
    {
        return reinterpret_cast<FixedVector<T, N> *>(arr);
    }

    //! Convert a array pointer to vector pointer
    static const FixedVector *Cast(const T arr[N])
    {
        return reinterpret_cast<const FixedVector<T, N> *>(arr);
    }

private:
    //! Data member
    T _data[N];
};

template <size_t N>
using FloatFixedVector = FixedVector<float, N>;

template <size_t N>
using DoubleFixedVector = FixedVector<double, N>;

template <size_t N>
using Int8FixedVector = FixedVector<int8_t, N>;

template <size_t N>
using Uint8FixedVector = FixedVector<uint8_t, N>;

template <size_t N>
using Int16FixedVector = FixedVector<int16_t, N>;

template <size_t N>
using Uint16FixedVector = FixedVector<uint16_t, N>;

template <size_t N>
using Int32FixedVector = FixedVector<int32_t, N>;

template <size_t N>
using Uint32FixedVector = FixedVector<uint32_t, N>;

template <size_t N>
using Int64FixedVector = FixedVector<int64_t, N>;

template <size_t N>
using Uint64FixedVector = FixedVector<uint64_t, N>;

} // namespace mercury

#endif // __MERCURY_UTILITY_FIXED_VECTOR_H__
