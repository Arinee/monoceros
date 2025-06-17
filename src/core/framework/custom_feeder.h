/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     custom_feeder.h
 *   \author   qiuming@xiaohongshu.com
 *   \date     Feb 2019
 *   \version  1.0.0
 *   \brief    Interface of Custom filter
 */

#ifndef __MERCURY_CUSTOM_FEEDER_H__
#define __MERCURY_CUSTOM_FEEDER_H__

#include "src/core/common/common.h"
#include <stdint.h>
#include <utility>

MERCURY_NAMESPACE_BEGIN(core);

class CustomFeeder {
public:
    CustomFeeder() 
    {}

    CustomFeeder(const CustomFeeder &rhs)
        :_feeder(rhs._feeder)
    {}

    CustomFeeder(CustomFeeder &&rhs)
        :_feeder(std::forward<decltype(_feeder)>(rhs._feeder))
    {}

    CustomFeeder & operator=(const CustomFeeder &rhs)
    {
        if (this == &rhs) {
            return *this;
        }
        _feeder = rhs._feeder;
        return *this;
    }

    CustomFeeder & operator=(CustomFeeder &&rhs)
    {
        if (this == &rhs) {
            return *this;
        }
        _feeder = std::forward<decltype(_feeder)>(rhs._feeder);
        return *this;
    }

    /*
     * -1 means EOF
     */
    key_t operator()(void) const
    {
        return _feeder();
    }

    template<typename T>
    void set(const T &func)
    {
        _feeder = func;
    }

    template<typename T>
    void set(T &&func)
    {
        _feeder = std::forward<T>(func);
    }

    bool isValid(void) const 
    {
        return (!!_feeder);
    }

private:
    std::function<key_t(void)> _feeder;
};

class CustomDocFeeder {
public:
    CustomDocFeeder() 
    {}

    CustomDocFeeder(const CustomDocFeeder &rhs)
        :_feeder(rhs._feeder)
    {}

    CustomDocFeeder(CustomDocFeeder &&rhs)
        :_feeder(std::forward<decltype(_feeder)>(rhs._feeder))
    {}

    CustomDocFeeder & operator=(const CustomDocFeeder &rhs)
    {
        if (this == &rhs) {
            return *this;
        }
        _feeder = rhs._feeder;
        return *this;
    }

    CustomDocFeeder & operator=(CustomDocFeeder &&rhs)
    {
        if (this == &rhs) {
            return *this;
        }
        _feeder = std::forward<decltype(_feeder)>(rhs._feeder);
        return *this;
    }

    /*
     * -1 means EOF
     */
    docid_t operator()(void) const
    {
        return _feeder();
    }

    template<typename T>
    void set(const T &func)
    {
        _feeder = func;
    }

    template<typename T>
    void set(T &&func)
    {
        _feeder = std::forward<T>(func);
    }

    bool isValid(void) const 
    {
        return (!!_feeder);
    }

private:
    std::function<docid_t(void)> _feeder;
};

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_CUSTOM_FEEDER_H__
