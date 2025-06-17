/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cube.h
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Cube
 */

#ifndef __MERCURY_UTILITY_CUBE_H__
#define __MERCURY_UTILITY_CUBE_H__

#include <algorithm>
#include <type_traits>
#include <typeinfo>

namespace mercury {
namespace internal {

/*! Cube Policy
 */
struct CubePolicy
{
    //! Assign `src` to `dst`
    virtual void assign(const void *src, void **dst) = 0;

    //! Cleanup value
    virtual void cleanup(void *val) = 0;

    //! Clone value of `src` to `dst`
    virtual void clone(void *const *src, void **dst) = 0;

    //! Move `src` to `dst`
    virtual void move(void *src, void **dst) = 0;

    //! Retrieve size
    virtual size_t size(void) const = 0;

    //! Retrieve type information
    virtual const std::type_info &type(void) const = 0;

    //! Retrieve value
    virtual void *value(void **src) = 0;

    //! Retrieve value
    virtual const void *value(void *const *src) const = 0;
};

/*! Small Cube Policy
 */
template <typename T>
struct SmallCubePolicy : public CubePolicy
{
    //! Assign `src` to `dst`
    void assign(const void *src, void **dst)
    {
        new (dst) T(*reinterpret_cast<const T *>(src));
    }

    //! Cleanup value
    void cleanup(void *) {}

    //! Clone value of `src` to `dst`
    void clone(void *const *src, void **dst)
    {
        *dst = *src;
    }

    //! Move `src` to `dst`
    void move(void *src, void **dst)
    {
        new (dst) T(std::move(*reinterpret_cast<T *>(src)));
    }

    //! Retrieve size
    size_t size(void) const
    {
        return sizeof(T);
    }

    //! Retrieve type information
    const std::type_info &type(void) const
    {
        return typeid(T);
    }

    //! Retrieve value
    void *value(void **src)
    {
        return reinterpret_cast<void *>(src);
    }

    //! Retrieve value
    const void *value(void *const *src) const
    {
        return reinterpret_cast<const void *>(src);
    }
};

/*! Large Cube Policy
 */
template <typename T>
struct LargeCubePolicy : public CubePolicy
{
    //! Assign `src` to `dst`
    void assign(const void *src, void **dst)
    {
        *dst = new T(*reinterpret_cast<const T *>(src));
    }

    //! Cleanup value
    void cleanup(void *val)
    {
        delete (reinterpret_cast<T *>(val));
    }

    //! Clone value of `src` to `dst`
    void clone(void *const *src, void **dst)
    {
        *dst = new T(**reinterpret_cast<T *const *>(src));
    }

    //! Move `src` to `dst`
    void move(void *src, void **dst)
    {
        *dst = new T(std::move(*reinterpret_cast<T *>(src)));
    }

    //! Retrieve size
    size_t size(void) const
    {
        return sizeof(T);
    }

    //! Retrieve type information
    const std::type_info &type(void) const
    {
        return typeid(T);
    }

    //! Retrieve value
    void *value(void **src)
    {
        return *src;
    }

    //! Retrieve value
    const void *value(void *const *src) const
    {
        return *src;
    }
};

/*! Policy Selector
 */
template <typename T>
struct PolicySelector
{
    typedef LargeCubePolicy<T> Type;
};

/*! Policy Selector
 */
template <typename T>
struct PolicySelector<T *>
{
    typedef SmallCubePolicy<T *> Type;
};

//! Policy Selector for `bool` type
template <>
struct PolicySelector<bool>
{
    typedef SmallCubePolicy<bool> Type;
};

//! Policy Selector for `char` type
template <>
struct PolicySelector<
    typename std::enable_if<!std::is_same<char, signed char>::value &&
                                !std::is_same<char, unsigned char>::value,
                            char>::type>
{
    typedef SmallCubePolicy<char> Type;
};

//! Policy Selector for `signed char` type
template <>
struct PolicySelector<signed char>
{
    typedef SmallCubePolicy<signed char> Type;
};

//! Policy Selector for `unsigned char` type
template <>
struct PolicySelector<unsigned char>
{
    typedef SmallCubePolicy<unsigned char> Type;
};

//! Policy Selector for `signed short` type
template <>
struct PolicySelector<signed short>
{
    typedef SmallCubePolicy<signed short> Type;
};

//! Policy Selector for `unsigned short` type
template <>
struct PolicySelector<unsigned short>
{
    typedef SmallCubePolicy<unsigned short> Type;
};

//! Policy Selector for `signed int` type
template <>
struct PolicySelector<signed int>
{
    typedef SmallCubePolicy<signed int> Type;
};

//! Policy Selector for `unsigned int` type
template <>
struct PolicySelector<unsigned int>
{
    typedef SmallCubePolicy<unsigned int> Type;
};

//! Policy Selector for `signed long` type
template <>
struct PolicySelector<typename std::enable_if<
    sizeof(signed long) <= sizeof(void *), signed long>::type>
{
    typedef SmallCubePolicy<signed long> Type;
};

//! Policy Selector for `unsigned long` type
template <>
struct PolicySelector<typename std::enable_if<
    sizeof(unsigned long) <= sizeof(void *), unsigned long>::type>
{
    typedef SmallCubePolicy<unsigned long> Type;
};

//! Policy Selector for `signed long long` type
template <>
struct PolicySelector<typename std::enable_if<
    sizeof(long long) <= sizeof(void *), signed long long>::type>
{
    typedef SmallCubePolicy<signed long long> Type;
};

//! Policy Selector for `unsigned long long` type
template <>
struct PolicySelector<typename std::enable_if<
    sizeof(unsigned long long) <= sizeof(void *), unsigned long long>::type>
{
    typedef SmallCubePolicy<unsigned long long> Type;
};

//! Policy Selector for `float` type
template <>
struct PolicySelector<float>
{
    typedef SmallCubePolicy<float> Type;
};

//! Policy Selector for `double` type
template <>
struct PolicySelector<
    typename std::enable_if<sizeof(double) <= sizeof(void *), double>::type>
{
    typedef SmallCubePolicy<double> Type;
};

//! Fixed underlying_type used with conditional
template <typename T, bool = std::is_enum<T>::value>
struct UnderlyingType
{
    typedef T Type;
};

//! Fixed underlying_type used with conditional
template <typename T>
struct UnderlyingType<T, true>
{
    typedef typename std::underlying_type<T>::type Type;
};

} // namespace internal

/*! Cube class
 */
class Cube
{
public:
    //! Constructor
    Cube(void) : _policy(Cube::Policy<Cube::EmptyPolicy>()), _object(nullptr) {}

    //! Constructor
    template <typename T>
    Cube(const T &rhs) : _policy(Cube::Policy<T>()), _object(nullptr)
    {
        _policy->assign(&rhs, &_object);
    }

    //! Constructor
    template <typename T, typename = typename std::enable_if<
                              !std::is_same<Cube &, T>::value &&
                              !std::is_same<T &, T>::value>::type>
    Cube(T &&rhs) : _policy(Cube::Policy<T>()), _object(nullptr)
    {
        _policy->move(&rhs, &_object);
    }

    //! Constructor
    Cube(const Cube &rhs) : _policy(rhs._policy), _object(nullptr)
    {
        _policy->clone(&rhs._object, &_object);
    }

    //! Constructor
    Cube(Cube &&rhs) : _policy(rhs._policy), _object(rhs._object)
    {
        rhs._policy = Cube::Policy<Cube::EmptyPolicy>();
        rhs._object = nullptr;
    }

    //! Destructor
    ~Cube(void)
    {
        _policy->cleanup(_object);
    }

    //! Assignment
    template <typename T>
    Cube &operator=(const T &rhs)
    {
        this->assign(rhs);
        return *this;
    }

    //! Assignment
    template <typename T, typename = typename std::enable_if<
                              !std::is_same<Cube &, T>::value &&
                              !std::is_same<T &, T>::value>::type>
    Cube &operator=(T &&rhs)
    {
        this->assign(std::forward<T>(rhs));
        return *this;
    }

    //! Assignment
    Cube &operator=(const Cube &rhs)
    {
        this->assign(rhs);
        return *this;
    }

    //! Assignment
    Cube &operator=(Cube &&rhs)
    {
        this->assign(std::forward<Cube>(rhs));
        return *this;
    }

    //! Retrieve object in original type
    template <typename T>
    operator T &()
    {
        return this->cast<T>();
    }

    //! Retrieve object in original type
    template <typename T>
    operator const T &() const
    {
        return this->cast<T>();
    }

    //! Assign content
    template <typename T>
    void assign(const T &rhs)
    {
        _policy->cleanup(_object);
        _policy = Cube::Policy<T>();
        _policy->assign(&rhs, &_object);
    }

    //! Assign content
    template <typename T, typename = typename std::enable_if<
                              !std::is_same<Cube &, T>::value &&
                              !std::is_same<T &, T>::value>::type>
    void assign(T &&rhs)
    {
        _policy->cleanup(_object);
        _policy = Cube::Policy<T>();
        _policy->move(&rhs, &_object);
    }

    //! Assign content from another Cube
    void assign(const Cube &rhs)
    {
        _policy->cleanup(_object);
        _policy = rhs._policy;
        _policy->clone(&rhs._object, &_object);
    }

    //! Assign content from another Cube
    void assign(Cube &&rhs)
    {
        if (this != &rhs) {
            _policy->cleanup(_object);
            _policy = rhs._policy;
            _object = rhs._object;
            rhs._policy = Cube::Policy<Cube::EmptyPolicy>();
            rhs._object = nullptr;
        }
    }

    //! Swap the content with another Cube
    Cube &swap(Cube &rhs)
    {
        std::swap(_policy, rhs._policy);
        std::swap(_object, rhs._object);
        return *this;
    }

    //! Cast to the original type
    template <typename T>
    T &cast(void)
    {
        if (_policy != Cube::Policy<T>()) {
            throw std::bad_cast();
        }
        return *reinterpret_cast<T *>(_policy->value(&_object));
    }

    //! Cast to the original type
    template <typename T>
    const T &cast(void) const
    {
        if (_policy != Cube::Policy<T>()) {
            throw std::bad_cast();
        }
        return *reinterpret_cast<const T *>(_policy->value(&_object));
    }

    //! Cast to the original type (unsafe)
    template <typename T>
    T &unsafe_cast(void)
    {
        return *reinterpret_cast<T *>(_policy->value(&_object));
    }

    //! Cast to the original type (unsafe)
    template <typename T>
    const T &unsafe_cast(void) const
    {
        return *reinterpret_cast<const T *>(_policy->value(&_object));
    }

    //! Test if the Cube is empty
    bool empty(void) const
    {
        return (_policy == Cube::Policy<Cube::EmptyPolicy>());
    }

    //! Reset Cube allocated memory
    void reset(void)
    {
        _policy->cleanup(_object);
        _policy = Cube::Policy<Cube::EmptyPolicy>();
        _object = nullptr;
    }

    //! Test if the Cube is compatible with another one
    bool compatible(const Cube &rhs) const
    {
        return (_policy == rhs._policy);
    }

    //! Test if the Cube is compatible with another one
    template <typename T>
    bool compatible(void) const
    {
        return (_policy == Cube::Policy<T>());
    }

    //! Retrieve size
    size_t size(void) const
    {
        return (!this->empty() ? _policy->size() : 0u);
    }

    //! Retrieve type information
    const std::type_info &type(void) const
    {
        return (!this->empty() ? _policy->type() : typeid(void));
    }

protected:
    /*! Empty Policy
     */
    struct EmptyPolicy
    {
    };

    //! Make a static policy object
    template <typename T>
    static internal::CubePolicy *MakePolicy(void)
    {
        static typename internal::PolicySelector<T>::Type policy;
        return (&policy);
    }

    //! Retrieve a static policy object
    template <typename T>
    static internal::CubePolicy *Policy(void)
    {
        return MakePolicy<typename internal::UnderlyingType<T>::Type>();
    }

private:
    //! Members
    internal::CubePolicy *_policy;
    void *_object;
};

} // namespace mercury

#endif // __MERCURY_UTILITY_CUBE_H__
