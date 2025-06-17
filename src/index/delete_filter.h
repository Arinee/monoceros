/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     delete_filter.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of delete filter
 */

#ifndef __MERCURY_DELETE_FILTER_H__
#define __MERCURY_DELETE_FILTER_H__

#include "framework/index_framework.h"
#include "index/index.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

class DeleteFilter 
{
public:
    typedef std::shared_ptr<DeleteFilter> Pointer;
public:
    DeleteFilter(Index *index)
        : _delMap(index->getDelMap())
    {}

    bool deleted(docid_t docid) const
    {
        return _delMap->test(docid);
    }

    bool empty(void) const 
    {
        return _delMap->testNone();
    }

private:
    const BitsetHelper *_delMap;
};

} // namespace mercury

#endif // __MERCURY_DELETE_FILTER_H__
