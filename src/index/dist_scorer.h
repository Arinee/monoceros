/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     dist_scorer.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of dist scorer
 */

#ifndef __MERCURY_DIST_SCORER_H__
#define __MERCURY_DIST_SCORER_H__

#include "framework/index_framework.h"
#include "index.h"
#include "index_scorer.h"
#include "general_search_context.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

class DistScorer : public IndexScorer
{
public:
    typedef std::shared_ptr<DistScorer> Pointer;
public:
    DistScorer() 
        : index_(nullptr)
    { }
    DistScorer(const DistScorer &scorer)
        : index_meta_(scorer.index_meta_)
        , index_(scorer.index_)
    { }

    virtual ~DistScorer() override
    {
    }
    virtual int Init(Index *index) override
    {
        if (!index) {
            return -1;
        }
        index_ = index;
        index_meta_ = *index_->get_index_meta();
        return 0;
    }
    virtual IndexScorer::Pointer Clone() override
    {
        return IndexScorer::Pointer(new DistScorer(*this));
    }
    virtual int ProcessQuery(const void * /*query*/, size_t /*bytes*/, GeneralSearchContext *context) override
    {
        if (context->getSearchMethod() != IndexDistance::kMethodUnknown) {
            index_meta_.setMethod(context->getSearchMethod());
        }
        return 0;
    }
    virtual float Score(const void *query, size_t /*bytes*/, docid_t docid) override
    {
        const void* feature = index_->getFeature(docid);
        return index_meta_.distance(query, feature);
    }
private:
    IndexMeta index_meta_;
    Index *index_;
};

} // namespace mercury

#endif // __MERCURY_DIST_SCORER_H__
