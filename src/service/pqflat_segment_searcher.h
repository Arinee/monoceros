/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     pqflat_segment_searcher.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of mercury PqflatSegmentSearcher
 */

#ifndef __MERCURY_PQFLAT_SEGMENT_SEARCHER_H__
#define __MERCURY_PQFLAT_SEGMENT_SEARCHER_H__

#include "framework/index_framework.h"
#include "index/index_pqflat.h"
#include "index/base_index_provider.h"
#include "index/delete_filter.h"
#include "index/pq_dist_scorer.h"
#include "utils/my_heap.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury {

class PqflatSegmentSearcher {
public:
    typedef std::shared_ptr<PqflatSegmentSearcher> Pointer;
public:
    PqflatSegmentSearcher(size_t sid) 
        :segid(sid),
        pqCodeProfile(nullptr),
        query(nullptr),
        bytes(0),
        context(nullptr)

    {} 
    virtual ~PqflatSegmentSearcher() 
    {}

    virtual void seekAndPush(QueryDistanceMatrix* qdm, MyHeap<DistNode>& result);
public:
    size_t segid;
    size_t docNum;
    ArrayProfile* pqCodeProfile;
    const void* query;
    size_t bytes;
    GeneralSearchContext* context;
    DeleteFilter* deleteFilter;
};

class PqflatSegmentSearcherWithFilter : public PqflatSegmentSearcher {
public:
    typedef std::shared_ptr<PqflatSegmentSearcherWithFilter> Pointer;
public:
    PqflatSegmentSearcherWithFilter(size_t sid) 
        :PqflatSegmentSearcher(sid),
         customFilter(nullptr),
         primaryKeyProfile(nullptr)
    {} 
    virtual ~PqflatSegmentSearcherWithFilter() override
    {}

    virtual void seekAndPush(QueryDistanceMatrix* qdm, MyHeap<DistNode> &result) override;
public:
    const CustomFilter* customFilter;
    ArrayProfile* primaryKeyProfile;
};

class PqflatSegmentSearcherWithFeeder : public PqflatSegmentSearcher {
public:
    typedef std::shared_ptr<PqflatSegmentSearcherWithFeeder> Pointer;
public:
    PqflatSegmentSearcherWithFeeder(size_t sid) 
        :PqflatSegmentSearcher(sid),
        customFeeder(nullptr),
        idMap(nullptr)
    {} 
    virtual ~PqflatSegmentSearcherWithFeeder() override
    {}

    virtual void seekAndPush(QueryDistanceMatrix* qdm, MyHeap<DistNode> &result) override;
public:
    const CustomFeeder* customFeeder;
    HashTable<uint64_t, docid_t>* idMap;
};

class PqflatSegSearcherFactory {
public:
    PqflatSegSearcherFactory() 
    {}
    int Init(const IndexParams &params);
    int Load(BaseIndexProvider *indexProvider);
    int Cleanup(void);
    int Unload(void);
    PqflatSegmentSearcher::Pointer Make(size_t segid, const void* /*query*/, size_t /*bytes*/,
            GeneralSearchContext* context);
private:
    std::vector<Index::Pointer> _indexes;
    std::vector<DeleteFilter::Pointer> _deleteFilters;
};

}; //namespace mercury


#endif //__MERCURY_PQFLAT_SEGMENT_SEARCHER_H__
