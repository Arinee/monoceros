/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     custom_retriever.h
 *   \author   jiaolong@xiaohongshu.com
 *   \date     Feb 2023
 *   \version  1.0.0
 *   \brief    Interface of deletion map retriever
 */

#ifndef __MERCURY_DELETION_MAP_RETRIEVER_H__
#define __MERCURY_DELETION_MAP_RETRIEVER_H__

#include "src/core/common/common.h"
#include <stdint.h>
#include <utility>

MERCURY_NAMESPACE_BEGIN(core);

class DeletionMapRetriever
{
public:
    DeletionMapRetriever() {}

    DeletionMapRetriever(const DeletionMapRetriever &rhs)
        : _retriever(rhs._retriever)
    {
    }

    DeletionMapRetriever(DeletionMapRetriever &&rhs)
        : _retriever(std::forward<decltype(_retriever)>(rhs._retriever))
    {
    }

    DeletionMapRetriever &operator=(const DeletionMapRetriever &rhs)
    {
        if (this == &rhs) {
            return *this;
        }
        _retriever = rhs._retriever;
        return *this;
    }

    DeletionMapRetriever &operator=(DeletionMapRetriever &&rhs)
    {
        if (this == &rhs) {
            return *this;
        }
        _retriever = std::forward<decltype(_retriever)>(rhs._retriever);
        return *this;
    }

    bool operator()(docid_t docid) const
    {
        return _retriever(docid);
    }

    template <typename T>
    void set(const T &func)
    {
        _retriever = func;
    }

    template <typename T>
    void set(T &&func)
    {
        _retriever = std::forward<T>(func);
    }

    bool isValid(void) const
    {
        return (!!_retriever);
    }

private:
    std::function<bool(docid_t)> _retriever;
};

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_DELETION_MAP_RETRIEVER_H__
