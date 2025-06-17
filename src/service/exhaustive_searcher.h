/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     exhaustive_searcher.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of mercury ExhaustiveSearcher
 */

#ifndef __MERCURY_EXHAUSTIVE_SEARCHER_H__
#define __MERCURY_EXHAUSTIVE_SEARCHER_H__

#include "framework/index_framework.h"
#include "index/base_index_provider.h"
#include "index/orig_dist_scorer.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

/* 
 * Only for exhuasitve search
 */
class ExhaustiveSearcher
{
public:
    typedef std::shared_ptr<ExhaustiveSearcher> Pointer;
public:
    ExhaustiveSearcher();
    ~ExhaustiveSearcher();
    //! Init from params
    int Init(const IndexParams &params);
    //! Cleanup searcher
    int Cleanup(void);
    //! Load from index provider
    int Load(BaseIndexProvider *indexProvider);
    //! Unload search hanlder
    int Unload();
    //! search by query
    int Search(const void* query, size_t bytes, size_t topk, GeneralSearchContext* context);
private: 
    std::vector<OrigDistScorer::Factory> _scorerFactories;
    size_t _segNum;
    BaseIndexProvider *_indexProvider;
};

} // namespace mercury

#endif // __MERCURY_EXHAUSTIVE_SEARCHER_H__
