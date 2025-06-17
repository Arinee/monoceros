/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     ivfflat_searcher.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of mercury ivfflat searcher
 */

#ifndef __MERCURY_IVFFLAT_SEARCHER_H__
#define __MERCURY_IVFFLAT_SEARCHER_H__

#include "framework/index_framework.h"
#include "index/base_index_provider.h"
#include "ivfflat_segment_searcher.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

class IvfflatSearcher
{
public:
    typedef std::shared_ptr<IvfflatSearcher> Pointer;
public:
    IvfflatSearcher();
    ~IvfflatSearcher();
    //! Init from params
    int Init(const IndexParams &params);
    //! Cleanup searcher
    int Cleanup(void);
    //! Load from index provider
    int Load(IvfFlatIndexProvider *indexProvider);
    //! Unload search hanlder
    int Unload();
    //! search by query
    int Search(const void* query, size_t bytes, size_t topk, GeneralSearchContext* context);
    //! exhaustive search by query
    int ExhaustiveSearch(const void* query, size_t bytes, size_t topk, GeneralSearchContext* context);

private:
    IvfflatSegSearcherFactory _segmentFactory;
    size_t _segNum;
    IndexMeta _indexMeta;
    IvfFlatIndexProvider *_indexProvider;
};

} // namespace mercury

#endif // __MERCURY_IVFFLAT_SEARCHER_H__
