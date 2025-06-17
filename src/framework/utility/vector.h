/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     vector.h
 *   \author   Hechong.xyf
 *   \date     Jul 2018
 *   \version  1.0.0
 *   \brief    Interface of Vector adapter
 */

#ifndef __MERCURY_UTILITY_VECTOR_H__
#define __MERCURY_UTILITY_VECTOR_H__

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace mercury {

/*! Vector Adapter
 */
#if __GNUG__ && __GNUC__ < 5
template <typename T, typename TBase = std::string,
          typename = typename std::enable_if<__has_trivial_copy(T)>::type>
#else
template <typename T, typename TBase = std::string,
          typename = typename std::enable_if<
              std::is_trivially_copyable<T>::value>::type>
#endif
class Vector : public TBase
{
public:
    typedef T value_type;
    typedef value_type *iterator;
    typedef const value_type *const_iterator;
    typedef value_type *reverse_iterator;
    typedef const value_type *const_reverse_iterator;

    //! Constructor
    Vector(void) : TBase() {}

    //! Constructor
    Vector(size_t count) : TBase()
    {
        this->resize(count);
    }

    //! Constructor
    Vector(size_t count, const value_type &val) : TBase()
    {
        this->resize(count, val);
    }

    //! Constructor
    Vector(const Vector &rhs) : TBase(rhs) {}

    //! Constructor
    Vector(Vector &&rhs) : TBase(std::move(rhs)) {}

    //! Constructor
    Vector(const TBase &rhs) : TBase(rhs)
    {
        if (TBase::size() % sizeof(T) != 0) {
            throw std::length_error("Unmatched length");
        }
    }

    //! Constructor
    Vector(TBase &&rhs) : TBase(std::move(rhs))
    {
        if (TBase::size() % sizeof(T) != 0) {
            throw std::length_error("Unmatched length");
        }
    }

    //! Constructor
    Vector(std::initializer_list<value_type> il) : TBase()
    {
        for (const auto &it : il) {
            TBase::append(reinterpret_cast<const char *>(&it),
                          sizeof(value_type));
        }
    }

    //! Assignment
    Vector &operator=(const Vector &rhs)
    {
        TBase::operator=(static_cast<const TBase &>(rhs));
        return *this;
    }

    //! Assignment
    Vector &operator=(Vector &&rhs)
    {
        TBase::operator=(std::move(static_cast<TBase &&>(rhs)));
        return *this;
    }

    //! Assignment
    Vector &operator=(const TBase &rhs)
    {
        TBase::operator=(rhs);
        return *this;
    }

    //! Assignment
    Vector &operator=(TBase &&rhs)
    {
        TBase::operator=(std::move(rhs));
        return *this;
    }

    //! Overloaded operator []
    value_type &operator[](size_t i)
    {
        return *(this->data() + i);
    }

    //! Overloaded operator []
    const value_type &operator[](size_t i) const
    {
        return *(this->data() + i);
    }

    //! Appends a copy of value
    Vector &append(const value_type &val)
    {
        TBase::append(reinterpret_cast<const char *>(&val), sizeof(value_type));
        return *this;
    }

    //! Append a copy of value
    void append(std::initializer_list<value_type> il)
    {
        for (const auto &it : il) {
            TBase::append(reinterpret_cast<const char *>(&it),
                          sizeof(value_type));
        }
    }

    //! Assign content to vector
    void assign(const value_type *vec, size_t len)
    {
        TBase::assign(reinterpret_cast<const char *>(vec),
                      len * sizeof(value_type));
    }

    //! Assign content to vector
    void assign(size_t n, const value_type &val)
    {
        this->clear();
        this->resize(n, val);
    }

    //! Assign content to vector
    void assign(std::initializer_list<value_type> il)
    {
        this->clear();
        for (const auto &it : il) {
            TBase::append(reinterpret_cast<const char *>(&it),
                          sizeof(value_type));
        }
    }

    //! Retrieve element
    value_type &at(size_t i)
    {
        return *(this->data() + i);
    }

    //! Retrieve element
    const value_type &at(size_t i) const
    {
        return *(this->data() + i);
    }

    //! Access last element
    value_type &back(void)
    {
        return *(this->rbegin());
    }

    //! Access last element
    const value_type &back(void) const
    {
        return *(this->rbegin());
    }

    //! Retrieve iterator to beginning
    iterator begin(void)
    {
        return this->data();
    }

    //! Retrieve iterator to beginning
    const_iterator begin(void) const
    {
        return this->data();
    }

    //! Retrieve size of allocated storage
    size_t capacity(void) const
    {
        return (TBase::capacity() / sizeof(value_type));
    }

    //! Clear the vector
    void clear(void)
    {
        TBase::clear();
    }

    //! Retrieve pointer of data
    value_type *data(void)
    {
        return const_cast<value_type *>(
            reinterpret_cast<const value_type *>(TBase::data()));
    }

    //! Retrieve pointer of data
    const value_type *data(void) const
    {
        return reinterpret_cast<const value_type *>(TBase::data());
    }

    //! Test if vector is empty
    bool empty(void) const
    {
        return TBase::empty();
    }

    //! An iterator to the past-the-end
    iterator end(void)
    {
        return (this->data() + this->size());
    }

    //! An iterator to the past-the-end
    const_iterator end(void) const
    {
        return (this->data() + this->size());
    }

    //! Access first element
    value_type &front(void)
    {
        return *(this->begin());
    }

    //! Access first element
    const value_type &front(void) const
    {
        return *(this->begin());
    }

    //! Retrieve reverse iterator to reverse beginning
    reverse_iterator rbegin(void)
    {
        return (this->data() + this->size() - 1);
    }

    //! Retrieve reverse iterator to reverse beginning
    const_reverse_iterator rbegin(void) const
    {
        return (this->data() + this->size() - 1);
    }

    //! Retrieve reverse iterator to reverse end
    reverse_iterator rend(void)
    {
        return (this->data() - 1);
    }

    //! Retrieve reverse iterator to reverse end
    const_reverse_iterator rend(void) const
    {
        return (this->data() - 1);
    }

    //! Request a change in capacity
    void reserve(size_t n)
    {
        TBase::reserve(n * sizeof(value_type));
    }

    //! Resize the vector to a length of n elements
    void resize(size_t n)
    {
        TBase::resize(n * sizeof(value_type));
    }

    //! Resize the vector to a length of n elements
    void resize(size_t n, const value_type &val)
    {
        size_t count = this->size();

        TBase::resize(n * sizeof(value_type));
        for (size_t i = count; i < n; ++i) {
            *(this->data() + i) = val;
        }
    }

    //! Retrieve size of vector
    size_t size(void) const
    {
        return (TBase::size() / sizeof(value_type));
    }

    //! Swap vector values
    void swap(Vector &vec)
    {
        TBase::swap(static_cast<TBase &>(vec));
    }
};

/*! Binary Vector Adapter
 */
template <typename T, typename TBase = std::string,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
class BinaryVector : public TBase
{
public:
    //! const_iterator of Binary Vector
    class const_iterator
    {
    public:
        //! Constructor
        const_iterator(void) : _i(0), _arr(nullptr) {}

        //! Constructor
        const_iterator(const void *buf, size_t i)
            : _i(i), _arr(reinterpret_cast<const uint8_t *>(buf))
        {
        }

        //! Equality
        bool operator==(const const_iterator &rhs) const
        {
            return (_i == rhs._i);
        }

        //! No equality
        bool operator!=(const const_iterator &rhs) const
        {
            return (_i != rhs._i);
        }

        //! Increment (Prefix)
        const_iterator &operator++()
        {
            ++_i;
            return *this;
        }

        //! Increment (Suffix)
        const_iterator operator++(int)
        {
            const_iterator tmp = *this;
            ++_i;
            return tmp;
        }

        //! Decrement (Prefix)
        const_iterator &operator--()
        {
            --_i;
            return *this;
        }

        //! Decrement (Suffix)
        const_iterator operator--(int)
        {
            const_iterator tmp = *this;
            --_i;
            return tmp;
        }

        //! operator "+="
        const_iterator &operator+=(size_t offset)
        {
            _i += offset;
            return *this;
        }

        //! operator "-="
        const_iterator &operator-=(size_t offset)
        {
            _i -= offset;
            return *this;
        }

        //! Indirection (eg. *iter)
        bool operator*() const
        {
            return ((_arr[_i >> 3] & (1u << (_i & 7))) != 0);
        }

    private:
        size_t _i;
        const uint8_t *_arr;
    };

    typedef const_iterator const_reverse_iterator;
    typedef T value_type;

    //! Constructor
    BinaryVector(void) : TBase() {}

    //! Constructor
    BinaryVector(size_t count) : TBase()
    {
        this->resize(count);
    }

    //! Constructor
    BinaryVector(size_t count, bool val) : TBase()
    {
        this->resize(count, val);
    }

    //! Constructor
    BinaryVector(const BinaryVector &rhs) : TBase(rhs) {}

    //! Constructor
    BinaryVector(BinaryVector &&rhs) : TBase(std::move(rhs)) {}

    //! Constructor
    BinaryVector(const TBase &rhs) : TBase(rhs)
    {
        if (TBase::size() % sizeof(T) != 0) {
            throw std::length_error("Unmatched length");
        }
    }

    //! Constructor
    BinaryVector(TBase &&rhs) : TBase(std::move(rhs))
    {
        if (TBase::size() % sizeof(T) != 0) {
            throw std::length_error("Unmatched length");
        }
    }

    //! Constructor
    BinaryVector(std::initializer_list<bool> il) : TBase()
    {
        this->resize(il.size());

        size_t index = 0;
        uint8_t *arr = const_cast<uint8_t *>(
            reinterpret_cast<const uint8_t *>(TBase::data()));

        for (auto val : il) {
            if (val) {
                arr[index >> 3] |= (1u << (index & 7));
            }
            ++index;
        }
    }

    //! Assignment
    BinaryVector &operator=(const BinaryVector &rhs)
    {
        TBase::operator=(static_cast<const TBase &>(rhs));
        return *this;
    }

    //! Assignment
    BinaryVector &operator=(BinaryVector &&rhs)
    {
        TBase::operator=(std::move(static_cast<TBase &&>(rhs)));
        return *this;
    }

    //! Assignment
    BinaryVector &operator=(const TBase &rhs)
    {
        TBase::operator=(rhs);
        return *this;
    }

    //! Assignment
    BinaryVector &operator=(TBase &&rhs)
    {
        TBase::operator=(std::move(rhs));
        return *this;
    }

    //! Overloaded operator []
    bool operator[](size_t i) const
    {
        const uint8_t *arr = reinterpret_cast<const uint8_t *>(TBase::data());
        return ((arr[i >> 3] & (1u << (i & 7))) != 0);
    }

    //! Assign content to vector
    void assign(const bool *vec, size_t len)
    {
        this->clear();
        this->resize(len);

        uint8_t *arr = const_cast<uint8_t *>(
            reinterpret_cast<const uint8_t *>(TBase::data()));
        for (size_t i = 0; i < len; ++i) {
            bool val = vec[i];
            if (val) {
                arr[i >> 3] |= (1u << (i & 7));
            }
        }
    }

    //! Assign content to vector
    void assign(size_t n, bool val)
    {
        this->clear();
        this->resize(n, val);
    }

    //! Assign content to vector
    void assign(std::initializer_list<bool> il)
    {
        this->clear();
        this->resize(il.size());

        size_t index = 0;
        uint8_t *arr = const_cast<uint8_t *>(
            reinterpret_cast<const uint8_t *>(TBase::data()));
        for (auto val : il) {
            if (val) {
                arr[index >> 3] |= (1u << (index & 7));
            }
            ++index;
        }
    }

    //! Retrieve element
    bool at(size_t i) const
    {
        const uint8_t *arr = reinterpret_cast<const uint8_t *>(TBase::data());
        return ((arr[i >> 3] & (1u << (i & 7))) != 0);
    }

    //! Set a bit
    void set(size_t i)
    {
        uint8_t *arr = const_cast<uint8_t *>(
            reinterpret_cast<const uint8_t *>(TBase::data()));
        arr[i >> 3] |= (1u << (i & 7));
    }

    //! Reset a bit
    void reset(size_t i)
    {
        uint8_t *arr = const_cast<uint8_t *>(
            reinterpret_cast<const uint8_t *>(TBase::data()));
        arr[i >> 3] &= ~(1u << (i & 7));
    }

    //! Toggle a bit
    void flip(size_t i)
    {
        uint8_t *arr = const_cast<uint8_t *>(
            reinterpret_cast<const uint8_t *>(TBase::data()));
        arr[i >> 3] ^= (1u << (i & 7));
    }

    //! Access last element
    bool back(void) const
    {
        return this->at(this->size() - 1);
    }

    //! Retrieve const_iterator to beginning
    const_iterator begin(void) const
    {
        return const_iterator(this->data(), 0);
    }

    //! Retrieve size of allocated storage
    size_t capacity(void) const
    {
        return (TBase::capacity() << 3);
    }

    //! Clear the vector
    void clear(void)
    {
        TBase::clear();
    }

    //! Retrieve pointer of data
    value_type *data(void)
    {
        return const_cast<value_type *>(
            reinterpret_cast<const value_type *>(TBase::data()));
    }

    //! Retrieve pointer of data
    const value_type *data(void) const
    {
        return reinterpret_cast<const value_type *>(TBase::data());
    }

    //! Test if vector is empty
    bool empty(void) const
    {
        return TBase::empty();
    }

    //! An const_iterator to the past-the-end
    const_iterator end(void) const
    {
        return const_iterator(this->data(), this->size());
    }

    //! Access first element
    bool front(void) const
    {
        return this->at(0);
    }

    //! Retrieve reverse const_iterator to reverse beginning
    const_reverse_iterator rbegin(void) const
    {
        return const_reverse_iterator(this->data(), this->size() - 1);
    }

    //! Retrieve reverse const_iterator to reverse end
    const_reverse_iterator rend(void) const
    {
        return const_reverse_iterator(this->data(), -1);
    }

    //! Request a change in capacity
    void reserve(size_t n)
    {
        TBase::reserve((n + (sizeof(value_type) << 3) - 1) /
                       (sizeof(value_type) << 3) * sizeof(value_type));
    }

    //! Resize the vector to a length of n elements
    void resize(size_t n)
    {
        TBase::resize((n + (sizeof(value_type) << 3) - 1) /
                      (sizeof(value_type) << 3) * sizeof(value_type));
    }

    //! Resize the vector to a length of n elements
    void resize(size_t n, bool val)
    {
        TBase::resize((n + (sizeof(value_type) << 3) - 1) /
                          (sizeof(value_type) << 3) * sizeof(value_type),
                      val ? 0xff : 0);
    }

    //! Retrieve size of vector
    size_t size(void) const
    {
        return (TBase::size() << 3);
    }

    //! Swap vector values
    void swap(BinaryVector &vec)
    {
        TBase::swap(static_cast<TBase &>(vec));
    }
};

} // namespace mercury

#endif //__MERCURY_UTILITY_VECTOR_H__
