/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     pqflat_searcher.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of mercury PqflatSearcher
 */

#ifndef __MERCURY_PQFLAT_SEARCHER_H__
#define __MERCURY_PQFLAT_SEARCHER_H__

#include "framework/index_framework.h"
#include "index/base_index_provider.h"
#include "pqflat_segment_searcher.h"
#include "distance_refiner.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

const size_t PQ_SCAN_NUM = 3000ul;

class PqflatSearcher
{
public:
    typedef std::shared_ptr<PqflatSearcher> Pointer;
public:
    PqflatSearcher();
    ~PqflatSearcher();
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
    PqflatSegSearcherFactory _segSearcherFactory;
    DistanceRefiner::Factory _refinerFactory;
    size_t _defaultPqScanNum;
    size_t _segNum;
    IndexMeta _indexMeta;
    BaseIndexProvider* _indexProvider;
    CentroidResource* _centroidResource;
};

} // namespace mercury

#endif // __MERCURY_PQFLAT_SEARCHER_H__
