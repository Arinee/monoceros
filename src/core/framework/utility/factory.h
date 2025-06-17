/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     factory.h
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Factory
 */

#ifndef __MERCURY_UTILITY_FACTORY_H__
#define __MERCURY_UTILITY_FACTORY_H__

#include <cstring>
#include <functional>
#include <map>
#include <memory>

namespace mercury {

/*! Factory
 */
template <typename TBase>
class Factory
{
public:
    /*! Factory Register
     */
    template <typename TImpl, typename = typename std::enable_if<
                                  std::is_base_of<TBase, TImpl>::value>::type>
    struct Register
    {
        //! Constructor
        Register(const char *key)
        {
            Factory::Instance()->emplace(
                key, [] { return new (std::nothrow) TImpl(); });
        }
    };

    //! Produce an instance (c_ptr)
    static TBase *Make(const char *key)
    {
        return Factory::Instance()->produce(key);
    }

    //! Produce an instance (shared_ptr)
    static std::shared_ptr<TBase> MakeShared(const char *key)
    {
        return std::shared_ptr<TBase>(Factory::Make(key));
    }

    //! Produce an instance (unique_ptr)
    static std::unique_ptr<TBase> MakeUnique(const char *key)
    {
        return std::unique_ptr<TBase>(Factory::Make(key));
    }

    //! Test if the element is exist
    static bool Has(const char *key)
    {
        return Factory::Instance()->has(key);
    }

protected:
    //! Constructor
    Factory(void) : _map() {}

    //! Retrieve the singleton factory
    static Factory *Instance(void)
    {
        static Factory factory;
        return (&factory);
    }

    //! Inserts a new element into map
    template <typename TFunc>
    void emplace(const char *key, TFunc &&func)
    {
        _map.emplace(key, func);
    }

    //! Produce an instance
    TBase *produce(const char *key)
    {
        auto iter = _map.find(key);
        if (iter != _map.end()) {
            return iter->second();
        }
        return nullptr;
    }

    //! Test if the element is exist
    bool has(const char *key)
    {
        return (_map.find(key) != _map.end());
    }

private:
    //! Disable them
    Factory(const Factory &);
    Factory(Factory &&);
    Factory &operator=(const Factory &);

    /*! Key Comparer
     */
    struct KeyComparer
    {
        bool operator()(const char *lhs, const char *rhs) const
        {
            return (std::strcmp(lhs, rhs) < 0);
        }
    };

    //! Don't use variable buffer as key store.
    //! The key must be use a static buffer to store.
    std::map<const char *, std::function<TBase *()>, KeyComparer> _map;
};

//! Factory Register
#define MERCURY_FACTORY_REGISTER(__NAME__, __BASE__, __IMPL__, ...)            \
    static mercury::Factory<__BASE__>::Register<__IMPL__>                      \
        _##__IMPL__##_FactoryRegister(__NAME__, ##__VA_ARGS__)

} // namespace mercury

#endif // __MERCURY_UTILITY_FACTORY_H__
