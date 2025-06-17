/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_scorer.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of index scorer
 */

#ifndef __MERCURY_INDEX_SCORER_H__
#define __MERCURY_INDEX_SCORER_H__

#include "framework/index_framework.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

class Index;
class GeneralSearchContext;

class IndexScorer
{
public:
    typedef std::shared_ptr<IndexScorer> Pointer;
public:
    virtual ~IndexScorer() {}
    virtual int Init(Index *index) = 0;
    virtual IndexScorer::Pointer Clone() = 0;
    virtual int ProcessQuery(const void *query, size_t bytes, GeneralSearchContext *context) = 0;
    virtual float Score(const void *query, size_t bytes, docid_t docid) = 0;
};

} // namespace mercury

#endif // __MERCURY_INDEX_SCORER_H__
