/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     vector_service.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of posting iterator
 */

#ifndef __MERCURY_POSTING_ITERATOR_H__
#define __MERCURY_POSTING_ITERATOR_H__

#include <memory>
#include <string>
#include <vector>
#include "common/common_define.h"

namespace mercury 
{

class PostingIterator
{
public:
    typedef std::shared_ptr<PostingIterator> Pointer;
public:
    virtual ~PostingIterator() {}
    //! 
    virtual docid_t next() = 0;
    virtual bool finish() = 0;
};

} // namespace mercury

#endif // __MERCURY_POSTING_ITERATOR_H__
