/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     custom_filter.h
 *   \author   qiuming@xiaohongshu.com
 *   \date     Feb 2019
 *   \version  1.0.0
 *   \brief    Interface of Custom filter
 */

#ifndef __MERCURY_CUSTOM_FILTER_H__
#define __MERCURY_CUSTOM_FILTER_H__

#include "common/common_define.h"
#include <stdint.h>
#include <utility>

namespace mercury {

class CustomFilter {
public:
    CustomFilter() 
    {}

    CustomFilter(const CustomFilter &rhs)
        :_filter(rhs._filter)
    {}

    CustomFilter(CustomFilter &&rhs)
        :_filter(std::forward<decltype(_filter)>(rhs._filter))
    {}

    CustomFilter & operator=(const CustomFilter &rhs)
    {
        if (this == &rhs) {
            return *this;
        }
        _filter = rhs._filter;
        return *this;
    }

    CustomFilter & operator=(CustomFilter &&rhs)
    {
        if (this == &rhs) {
            return *this;
        }
        _filter = std::forward<decltype(_filter)>(rhs._filter);
        return *this;
    }

    bool operator()(key_t key) const
    {
        return _filter(key);
    }

    template<typename T>
    void set(const T &func)
    {
        _filter = func;
    }

    template<typename T>
    void set(T &&func)
    {
        _filter = std::forward<T>(func);
    }

    bool isValid(void) const 
    {
        return (!!_filter);
    }
private:
    std::function<bool(key_t)> _filter;
};

}; // namespace mercury

#endif // __MERCURY_CUSTOM_FILTER_H__
