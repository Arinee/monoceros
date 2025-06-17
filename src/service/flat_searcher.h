/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     flat_searcher.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of mercury FlatSearcher
 */

#ifndef __MERCURY_FLAT_SEARCHER_H__
#define __MERCURY_FLAT_SEARCHER_H__

#include "framework/index_framework.h"
#include "index/base_index_provider.h"
#include "flat_segment_searcher.h"
#include "distance_refiner.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

class FlatSearcher
{
public:
    typedef std::shared_ptr<FlatSearcher> Pointer;
public:
    FlatSearcher();
    ~FlatSearcher();
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
    FlatSegSearcherFactory _segSearcherFactory;
    size_t _segNum;
    BaseIndexProvider* _indexProvider;
};

} // namespace mercury

#endif // __MERCURY_FLAT_SEARCHER_H__
