/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     ivfflat_segment_searcher.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of mercury IvfflatSegmentSearcher
 */

#ifndef __MERCURY_IVFFLAT_SEGMENT_SEARCHER_H__
#define __MERCURY_IVFFLAT_SEGMENT_SEARCHER_H__

#include "framework/index_framework.h"
#include "index/ivfflat_index_provider.h"
#include "index/orig_dist_scorer.h"
#include "index/ivf_seeker.h"
#include "index/delete_filter.h"
#include "utils/my_heap.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury {

const float LEVEL_SCAN_RATIO = 0.05f;
const float COARSE_PROBE_RATIO = 0.01f;

class IvfflatSegmentSearcher {
public:
    typedef std::shared_ptr<IvfflatSegmentSearcher> Pointer;
public:
    explicit IvfflatSegmentSearcher(size_t sid)
        :segid(sid),
        featureProfile(nullptr),
        ivfSeeker(nullptr),
        query(nullptr),
        bytes(0),
        context(nullptr)
    {} 
    virtual ~IvfflatSegmentSearcher() = default;

    virtual void seekAndPush(MyHeap<DistNode>& result);
public:
    size_t segid;
    ArrayProfile* featureProfile;
    IvfSeeker* ivfSeeker;
    const void* query;
    size_t bytes;
    GeneralSearchContext* context;
    OrigDistScorer scorer;
    DeleteFilter* deleteFilter;
};

class IvfflatSegmentSearcherWithFilter : public IvfflatSegmentSearcher {
public:
    typedef std::shared_ptr<IvfflatSegmentSearcherWithFilter> Pointer;
public:
    explicit IvfflatSegmentSearcherWithFilter(size_t sid)
        :IvfflatSegmentSearcher(sid),
        customFilter(nullptr),
        primaryKeyProfile(nullptr)
    {} 
    ~IvfflatSegmentSearcherWithFilter() override = default;

    void seekAndPush(MyHeap<DistNode> &result) override;
public:
    const CustomFilter* customFilter;
    ArrayProfile* primaryKeyProfile;
};

class IvfflatSegmentSearcherWithFeeder : public IvfflatSegmentSearcher {
public:
    typedef std::shared_ptr<IvfflatSegmentSearcherWithFeeder> Pointer;
public:
    explicit IvfflatSegmentSearcherWithFeeder(size_t sid)
        :IvfflatSegmentSearcher(sid),
        customFeeder(nullptr),
        idMap(nullptr)
    {} 
    ~IvfflatSegmentSearcherWithFeeder() override = default;

    void seekAndPush(MyHeap<DistNode> &result) override;
public:
    const CustomFeeder* customFeeder;
    HashTable<uint64_t, docid_t>* idMap;
};

class IvfflatSegSearcherFactory {
public:
    IvfflatSegSearcherFactory() = default;
    int Init(const IndexParams &params);
    int Load(IvfFlatIndexProvider *indexProvider);
    int Cleanup(void);
    int Unload(void);
    IvfflatSegmentSearcher::Pointer Make(size_t segid, const void* /*query*/, size_t /*bytes*/,
            GeneralSearchContext* context);
private:
    std::vector<float> _defaultCoarseScanRatio;
    std::vector<Index::Pointer> _indexes;
    std::vector<IvfSeeker::Pointer> _ivfSeekers;
    std::vector<DeleteFilter::Pointer> _deleteFilters;
    std::vector<OrigDistScorer::Factory> _scorerFactories;
};

}; //namespace mercury

#endif //__MERCURY_IVFFLAT_SEGMENT_SEARCHER_H__
