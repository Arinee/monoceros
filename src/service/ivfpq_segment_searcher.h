/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     ivfpq_segment_searcher.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of mercury IvfpqSegmentSearcher
 */

#ifndef __MERCURY_IVFPQ_SEGMENT_SEARCHER_H__
#define __MERCURY_IVFPQ_SEGMENT_SEARCHER_H__

#include "framework/index_framework.h"
#include "index/ivfpq_index_provider.h"
#include "index/ivf_seeker.h"
#include "index/delete_filter.h"
#include "index/pq_dist_scorer.h"
#include "utils/my_heap.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury {

const float LEVEL_SCAN_RATIO = 0.05f;
const float COARSE_PROBE_RATIO = 0.01f;

class IvfpqSegmentSearcher {
public:
    typedef std::shared_ptr<IvfpqSegmentSearcher> Pointer;
public:
    IvfpqSegmentSearcher(size_t sid) 
        :segid(sid),
        pqCodeProfile(nullptr),
        ivfSeeker(nullptr),
        query(nullptr),
        bytes(0),
        context(nullptr)
    {} 
    virtual ~IvfpqSegmentSearcher() 
    {}

    virtual void seekAndPush(QueryDistanceMatrix* qdm, MyHeap<DistNode>& result);
public:
    size_t segid;
    ArrayProfile* pqCodeProfile;
    IvfSeeker* ivfSeeker;
    const void* query;
    size_t bytes;
    GeneralSearchContext* context;
    DeleteFilter* deleteFilter;
};

class IvfpqSegmentSearcherWithFilter : public IvfpqSegmentSearcher {
public:
    typedef std::shared_ptr<IvfpqSegmentSearcherWithFilter> Pointer;
public:
    IvfpqSegmentSearcherWithFilter(size_t sid) 
        :IvfpqSegmentSearcher(sid),
        customFilter(nullptr),
        primaryKeyProfile(nullptr)
    {} 
    virtual ~IvfpqSegmentSearcherWithFilter() override
    {}

    virtual void seekAndPush(QueryDistanceMatrix* qdm, MyHeap<DistNode> &result) override;
public:
    const CustomFilter* customFilter;
    ArrayProfile* primaryKeyProfile;
};

class IvfpqSegmentSearcherWithFeeder : public IvfpqSegmentSearcher {
public:
    typedef std::shared_ptr<IvfpqSegmentSearcherWithFeeder> Pointer;
public:
    IvfpqSegmentSearcherWithFeeder(size_t sid) 
        :IvfpqSegmentSearcher(sid),
        customFeeder(nullptr),
        idMap(nullptr)
    {} 
    virtual ~IvfpqSegmentSearcherWithFeeder() override
    {}

    virtual void seekAndPush(QueryDistanceMatrix* qdm, MyHeap<DistNode> &result) override;
public:
    const CustomFeeder* customFeeder;
    HashTable<uint64_t, docid_t>* idMap;
};

class IvfpqSegSearcherFactory {
public:
    IvfpqSegSearcherFactory() 
    {}
    int Init(const IndexParams &params);
    int Load(IvfpqIndexProvider *indexProvider);
    int Cleanup(void);
    int Unload(void);
    IvfpqSegmentSearcher::Pointer Make(size_t segid, const void* /*query*/, size_t /*bytes*/,
            GeneralSearchContext* context);
private:
    std::vector<float> _defaultCoarseScanRatio;
    std::vector<Index::Pointer> _indexes;
    std::vector<IvfSeeker::Pointer> _ivfSeekers;
    std::vector<DeleteFilter::Pointer> _deleteFilters;
};

}; //namespace mercury


#endif //__MERCURY_IVFPQ_SEGMENT_SEARCHER_H__
