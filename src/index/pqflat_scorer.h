/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     pq_scorer.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of pq scorer
 */

#ifndef __MERCURY_PQFLAT_SCORER_H__
#define __MERCURY_PQFLAT_SCORER_H__

#include "index_scorer.h"
#include "framework/index_framework.h"
#include "index_pqflat.h"
#include "general_search_context.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

class PqflatScorer : public IndexScorer
{
public:
    typedef std::shared_ptr<PqflatScorer> Pointer;
public:
    PqflatScorer() 
        : index_(nullptr)
    { }
    PqflatScorer(const PqflatScorer &scorer)
        : index_(scorer.index_)
    { }
    virtual ~PqflatScorer() override
    { }

    virtual int Init(Index *index) override
    {
        index_ = dynamic_cast<IndexPqflat*>(index);
        if (!index_) {
            return -1;
        }
        return 0;
    }
    virtual IndexScorer::Pointer Clone() override
    {
        return IndexScorer::Pointer(new PqflatScorer(*this));
    }
    virtual int ProcessQuery(const void * query, size_t /*bytes*/, GeneralSearchContext *context) override
    {
        // if qdm is empty, then create it once
        if (!context->getQueryDistanceMatrix()) {
            if (index_->InitQueryDistanceMatrix(query, context) != true) {
                return -1;
            }
        }
        qdm_ = context->getQueryDistanceMatrix().get();
        return 0;
    }
    virtual float Score(const void * /*query*/, size_t /*bytes*/, docid_t docid) override
    {
        const uint16_t * pq_code =  index_->getPqCode(docid);
        return IndexPqflat::calcScore(pq_code, qdm_);
    }
private:
    IndexPqflat* index_;
    QueryDistanceMatrix* qdm_;
};

} // namespace mercury

#endif // __MERCURY_PQFLAT_SCORER_H__
