/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     closure.h
 *   \author   Hechong.xyf
 *   \date     Jan 2018
 *   \version  1.0.5
 *   \brief    Interface of Mercury Utility Closure
 *   \detail   Construct a closure and run it at another time.
 *             All closure objects use the same running
 *             interfaces, but they can be constructed with
 *             different functions and parameters.
 *             The parameters will be saved into the closure
 *             objects, then passed to the callback functions
 *             when they are invoked.
 */

#ifndef __MERCURY_UTILITY_CLOSURE_H__
#define __MERCURY_UTILITY_CLOSURE_H__

#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>

namespace mercury {

/*! Callback closure
 */
struct Closure
{
    /*! Callback closure pointer
     */
    typedef std::shared_ptr<Closure> Pointer;

    //! Destructor
    virtual ~Closure(void) {}

    //! Run the closure
    virtual void run(void) = 0;

    //! Create callback closure
    template <typename R, typename... TParams, typename... TArgs>
    inline static Pointer New(R (*impl)(TParams...), TArgs &&... args);

    //! Create callback closure (non-parametric)
    template <typename R>
    inline static Pointer New(R (*impl)(void));

    //! Create callback closure
    template <typename R, typename T, typename... TParams, typename... TArgs>
    inline static Pointer New(T *obj, R (T::*impl)(TParams...),
                              TArgs &&... args);

    //! Create callback closure (constable)
    template <typename R, typename T, typename... TParams, typename... TArgs>
    inline static Pointer New(T *obj, R (T::*impl)(TParams...) const,
                              TArgs &&... args);

    //! Create callback closure (non-parametric)
    template <typename R, typename T>
    inline static Pointer New(T *obj, R (T::*impl)(void));

    //! Create callback closure (non-parametric, constable)
    template <typename R, typename T>
    inline static Pointer New(T *obj, R (T::*impl)(void) const);

    //! Create callback closure
    template <typename R, typename... TParams, typename... TArgs>
    inline static Pointer New(const std::function<R(TParams...)> &impl,
                              TArgs &&... args);

    //! Create callback closure
    template <typename R, typename... TParams, typename... TArgs>
    inline static Pointer New(std::function<R(TParams...)> &&impl,
                              TArgs &&... args);

    //! Create callback closure (non-parametric)
    template <typename R>
    inline static Pointer New(const std::function<R(void)> &impl);

    //! Create callback closure (non-parametric)
    template <typename R>
    inline static Pointer New(std::function<R(void)> &&impl);

protected:
    Closure(void){};
};

/*! Closure Functor
 */
template <typename TFunc, typename... TParams>
struct CallbackFunctor
{
    //! Tuple Index Maker
    template <size_t N, size_t... I>
    struct TupleIndexMaker : TupleIndexMaker<N - 1, N - 1, I...>
    {
    };

    //! Tuple Index
    template <size_t...>
    struct TupleIndex
    {
    };

    //! Tuple Index Maker (special)
    template <size_t... I>
    struct TupleIndexMaker<0, I...>
    {
        typedef TupleIndex<I...> Type;
    };

    //! Types redefinition
    typedef typename std::decay<TFunc>::type Type;
    typedef std::tuple<typename std::decay<TParams>::type...> TupleType;

    //! Run the callback function
    template <size_t... I>
    static void Run(Type &impl, TupleType &tuple, TupleIndex<I...>)
    {
        (impl)(std::forward<TParams>(std::get<I>(tuple))...);
    }

    //! Run the callback member function
    template <typename T, size_t... I>
    static void Run(T *obj, Type &impl, TupleType &tuple, TupleIndex<I...>)
    {
        (obj->*impl)(std::forward<TParams>(std::get<I>(tuple))...);
    }

    //! Run the callback function
    static void Run(Type &impl, TupleType &tuple)
    {
        CallbackFunctor::Run(
            impl, tuple, typename TupleIndexMaker<sizeof...(TParams)>::Type());
    }

    //! Run the callback member function
    template <typename T>
    static void Run(T *obj, Type &impl, TupleType &tuple)
    {
        CallbackFunctor::Run(
            obj, impl, tuple,
            typename TupleIndexMaker<sizeof...(TParams)>::Type());
    }
};

/*! Closure Functor (special)
 */
template <typename TFunc>
struct CallbackFunctor<TFunc, void>
{
    //! Types redefinition
    typedef typename std::decay<TFunc>::type Type;

    //! Run the callback function
    static void Run(Type &impl)
    {
        (impl)();
    }

    //! Run the callback member function
    template <typename T>
    static void Run(T *obj, Type &impl)
    {
        (obj->*impl)();
    }
};

/*! Class member callback (N parameters)
 */
template <typename T, typename TFunc, typename... TParams>
class Callback : public Closure
{
public:
    //! Constructor
    template <typename... TArgs,
              typename = typename std::enable_if<sizeof...(TArgs) ==
                                                 sizeof...(TParams)>::type>
    Callback(T *obj,
             const typename CallbackFunctor<TFunc, TParams...>::Type &impl,
             TArgs &&... args)
        : _obj(obj), _impl(impl), _tuple(std::forward<TArgs>(args)...)
    {
    }

    //! Constructor
    template <typename... TArgs,
              typename = typename std::enable_if<sizeof...(TArgs) ==
                                                 sizeof...(TParams)>::type>
    Callback(T *obj, typename CallbackFunctor<TFunc, TParams...>::Type &&impl,
             TArgs &&... args)
        : _obj(obj), _impl(std::move(impl)),
          _tuple(std::forward<TArgs>(args)...)
    {
    }

    //! Run the callback function
    virtual void run(void)
    {
        CallbackFunctor<TFunc, TParams...>::Run(_obj, _impl, _tuple);
    }

protected:
    //! Disable them
    Callback(void) = delete;
    Callback(const Callback &) = delete;
    Callback(Callback &&) = delete;
    Callback &operator=(const Callback &) = delete;

private:
    T *_obj;
    typename CallbackFunctor<TFunc, TParams...>::Type _impl;
    typename CallbackFunctor<TFunc, TParams...>::TupleType _tuple;
};

/*! Class member callback (non-parametric)
 */
template <typename T, typename TFunc>
class Callback<T, TFunc> : public Closure
{
public:
    //! Constructor
    Callback(T *obj, const typename CallbackFunctor<TFunc, void>::Type &impl)
        : _obj(obj), _impl(impl)
    {
    }

    //! Constructor
    Callback(T *obj, typename CallbackFunctor<TFunc, void>::Type &&impl)
        : _obj(obj), _impl(std::move(impl))
    {
    }

    //! Run the callback function
    virtual void run(void)
    {
        CallbackFunctor<TFunc, void>::Run(_obj, _impl);
    }

protected:
    //! Disable them
    Callback(void) = delete;
    Callback(const Callback &) = delete;
    Callback(Callback &&) = delete;
    Callback &operator=(const Callback &) = delete;

private:
    T *_obj;
    typename CallbackFunctor<TFunc, void>::Type _impl;
};

/*! Static member callback (N parameters)
 */
template <typename TFunc, typename... TParams>
class Callback<void, TFunc, TParams...> : public Closure
{
public:
    //! Constructor
    template <typename... TArgs,
              typename = typename std::enable_if<sizeof...(TArgs) ==
                                                 sizeof...(TParams)>::type>
    Callback(const typename CallbackFunctor<TFunc, TParams...>::Type &impl,
             TArgs &&... args)
        : _impl(impl), _tuple(std::forward<TArgs>(args)...)
    {
    }

    //! Constructor
    template <typename... TArgs,
              typename = typename std::enable_if<sizeof...(TArgs) ==
                                                 sizeof...(TParams)>::type>
    Callback(typename CallbackFunctor<TFunc, TParams...>::Type &&impl,
             TArgs &&... args)
        : _impl(std::move(impl)), _tuple(std::forward<TArgs>(args)...)
    {
    }

    //! Run the callback function
    virtual void run(void)
    {
        CallbackFunctor<TFunc, TParams...>::Run(_impl, _tuple);
    }

protected:
    //! Disable them
    Callback(void) = delete;
    Callback(const Callback &) = delete;
    Callback(Callback &&) = delete;
    Callback &operator=(const Callback &) = delete;

private:
    typename CallbackFunctor<TFunc, TParams...>::Type _impl;
    typename CallbackFunctor<TFunc, TParams...>::TupleType _tuple;
};

/*! Static member callback (non-parametric)
 */
template <typename TFunc>
class Callback<void, TFunc> : public Closure
{
public:
    //! Constructor
    Callback(const typename CallbackFunctor<TFunc, void>::Type &impl)
        : _impl(impl)
    {
    }

    //! Constructor
    Callback(typename CallbackFunctor<TFunc, void>::Type &&impl)
        : _impl(std::move(impl))
    {
    }

    //! Run the callback function
    virtual void run(void)
    {
        CallbackFunctor<TFunc, void>::Run(_impl);
    }

protected:
    //! Disable them
    Callback(void) = delete;
    Callback(const Callback &) = delete;
    Callback(Callback &&) = delete;
    Callback &operator=(const Callback &) = delete;

private:
    typename CallbackFunctor<TFunc, void>::Type _impl;
};

//! Create callback closure
template <typename R, typename... TParams, typename... TArgs>
Closure::Pointer Closure::New(R (*impl)(TParams...), TArgs &&... args)
{
    static_assert(sizeof...(TArgs) == sizeof...(TParams),
                  "Unmatched arguments");
    return std::make_shared<Callback<void, decltype(impl), TParams...>>(
        impl, std::forward<TArgs>(args)...);
}

//! Create callback closure (non-parametric)
template <typename R>
Closure::Pointer Closure::New(R (*impl)(void))
{
    return std::make_shared<Callback<void, decltype(impl)>>(impl);
}

//! Create callback closure
template <typename R, typename T, typename... TParams, typename... TArgs>
Closure::Pointer Closure::New(T *obj, R (T::*impl)(TParams...),
                              TArgs &&... args)
{
    static_assert(sizeof...(TArgs) == sizeof...(TParams),
                  "Unmatched arguments");
    return std::make_shared<Callback<T, decltype(impl), TParams...>>(
        obj, impl, std::forward<TArgs>(args)...);
}

//! Create callback closure (constable)
template <typename R, typename T, typename... TParams, typename... TArgs>
Closure::Pointer Closure::New(T *obj, R (T::*impl)(TParams...) const,
                              TArgs &&... args)
{
    static_assert(sizeof...(TArgs) == sizeof...(TParams),
                  "Unmatched arguments");
    return std::make_shared<Callback<T, decltype(impl), TParams...>>(
        obj, impl, std::forward<TArgs>(args)...);
}

//! Create callback closure (non-parametric)
template <typename R, typename T>
Closure::Pointer Closure::New(T *obj, R (T::*impl)(void))
{
    return std::make_shared<Callback<T, decltype(impl)>>(obj, impl);
}

//! Create callback closure (non-parametric, constable)
template <typename R, typename T>
Closure::Pointer Closure::New(T *obj, R (T::*impl)(void) const)
{
    return std::make_shared<Callback<T, decltype(impl)>>(obj, impl);
}

//! Create callback closure
template <typename R, typename... TParams, typename... TArgs>
Closure::Pointer Closure::New(const std::function<R(TParams...)> &impl,
                              TArgs &&... args)
{
    static_assert(sizeof...(TArgs) == sizeof...(TParams),
                  "Unmatched arguments");
    return std::make_shared<Callback<void, decltype(impl), TParams...>>(
        impl, std::forward<TArgs>(args)...);
}

//! Create callback closure
template <typename R, typename... TParams, typename... TArgs>
Closure::Pointer Closure::New(std::function<R(TParams...)> &&impl,
                              TArgs &&... args)
{
    static_assert(sizeof...(TArgs) == sizeof...(TParams),
                  "Unmatched arguments");
    return std::make_shared<Callback<void, decltype(impl), TParams...>>(
        std::move(impl), std::forward<TArgs>(args)...);
}

//! Create callback closure (non-parametric)
template <typename R>
Closure::Pointer Closure::New(const std::function<R(void)> &impl)
{
    return std::make_shared<Callback<void, decltype(impl)>>(impl);
}

//! Create callback closure (non-parametric)
template <typename R>
Closure::Pointer Closure::New(std::function<R(void)> &&impl)
{
    return std::make_shared<Callback<void, decltype(impl)>>(std::move(impl));
}

} // namespace mercury

#endif // __MERCURY_UTILITY_CLOSURE_H__
