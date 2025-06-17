/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     custom_retriever.h
 *   \author   qiuming@xiaohongshu.com
 *   \date     Feb 2019
 *   \version  1.0.0
 *   \brief    Interface of Custom attr retriever
 */

#ifndef __MERCURY_ATTR_RETRIEVER_H__
#define __MERCURY_ATTR_RETRIEVER_H__

#include "src/core/common/common.h"
#include <stdint.h>
#include <utility>

MERCURY_NAMESPACE_BEGIN(core);

class AttrRetriever {
public:
    AttrRetriever()
        : _length_fixed(false)
    {}

    AttrRetriever(const AttrRetriever &rhs)
        : _retriever(rhs._retriever)
        , _length_fixed(rhs._length_fixed)
    {}

    AttrRetriever(AttrRetriever &&rhs)
        : _retriever(std::forward<decltype(_retriever)>(rhs._retriever))
        , _length_fixed(rhs._length_fixed)
    {}

    AttrRetriever & operator=(const AttrRetriever &rhs)
    {
        if (this == &rhs) {
            return *this;
        }
        _retriever = rhs._retriever;
        _length_fixed = rhs._length_fixed;
        return *this;
    }

    AttrRetriever & operator=(AttrRetriever &&rhs)
    {
        if (this == &rhs) {
            return *this;
        }
        _retriever = std::forward<decltype(_retriever)>(rhs._retriever);
        _length_fixed = rhs._length_fixed;
        return *this;
    }

    bool operator()(docid_t docid, const void*& base) const
    {
        return _retriever(docid, base);
    }

    template<typename T>
    void set(const T &func)
    {
        _retriever = func;
    }

    template<typename T>
    void set(T &&func)
    {
        _retriever = std::forward<T>(func);
    }

    bool isValid(void) const 
    {
        return (!!_retriever);
    }

    void setLengthFixed() {
        _length_fixed = true;
    }

    bool isLengthFixed() {
        return _length_fixed;
    }
private:
    std::function<bool(docid_t, const void*&)> _retriever;
    bool _length_fixed;
};

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_AttrRetriever_H__
