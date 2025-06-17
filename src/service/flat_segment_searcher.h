/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     flat_segment_searcher.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of mercury FlatSegmentSearcher
 */

#ifndef __MERCURY_FLAT_SEGMENT_SEARCHER_H__
#define __MERCURY_FLAT_SEGMENT_SEARCHER_H__

#include "framework/index_framework.h"
#include "index/index.h"
#include "index/orig_dist_scorer.h"
#include "index/base_index_provider.h"
#include "index/delete_filter.h"
#include "utils/my_heap.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury {

class FlatSegmentSearcher {
public:
    typedef std::shared_ptr<FlatSegmentSearcher> Pointer;
public:
    FlatSegmentSearcher(size_t sid) 
        :segid(sid),
        query(nullptr),
        context(nullptr)

    {} 
    virtual ~FlatSegmentSearcher() 
    {}

    virtual void seekAndPush(MyHeap<DistNode>& result);
public:
    size_t segid;
    size_t docNum;
    const void* query;
    GeneralSearchContext* context;
    OrigDistScorer scorer;
    DeleteFilter* deleteFilter;
};

class FlatSegmentSearcherWithFilter : public FlatSegmentSearcher {
public:
    typedef std::shared_ptr<FlatSegmentSearcherWithFilter> Pointer;
public:
    FlatSegmentSearcherWithFilter(size_t sid) 
        : FlatSegmentSearcher(sid),
        customFilter(nullptr),
        primaryKeyProfile(nullptr)
    {} 
    virtual ~FlatSegmentSearcherWithFilter() override
    {}

    virtual void seekAndPush(MyHeap<DistNode> &result) override;
public:
    const CustomFilter *customFilter;
    ArrayProfile* primaryKeyProfile;
};

class FlatSegmentSearcherWithFeeder : public FlatSegmentSearcher {
public:
    typedef std::shared_ptr<FlatSegmentSearcherWithFeeder> Pointer;
public:
    FlatSegmentSearcherWithFeeder(size_t sid) 
        : FlatSegmentSearcher(sid),
        customFeeder(nullptr),
        idMap(nullptr)
    {} 
    virtual ~FlatSegmentSearcherWithFeeder() override
    {}

    virtual void seekAndPush(MyHeap<DistNode> &result) override;
public:
    const CustomFeeder *customFeeder;
    HashTable<uint64_t, docid_t> *idMap;
};

class FlatSegmentSearcherWithDocFeeder : public FlatSegmentSearcher {
public:
    typedef std::shared_ptr<FlatSegmentSearcherWithDocFeeder> Pointer;
public:
    FlatSegmentSearcherWithDocFeeder(size_t sid) 
        : FlatSegmentSearcher(sid),
        customDocFeeder(nullptr),
        idMap(nullptr)
    {} 
    virtual ~FlatSegmentSearcherWithDocFeeder() override
    {}

    virtual void seekAndPush(MyHeap<DistNode> &result) override;
public:
    const CustomDocFeeder *customDocFeeder;
    HashTable<uint64_t, docid_t> *idMap;
};

class FlatSegSearcherFactory {
public:
    FlatSegSearcherFactory() 
    {}
    int Init(const IndexParams &params);
    int Load(BaseIndexProvider *indexProvider);
    int Cleanup(void);
    int Unload(void);
    FlatSegmentSearcher::Pointer Make(size_t segid, const void* /*query*/, size_t /*bytes*/,
            GeneralSearchContext* context);
private:
    std::vector<Index::Pointer> _indexes;
    std::vector<DeleteFilter::Pointer> _deleteFilters;
    std::vector<OrigDistScorer::Factory> _scorerFactories;
};

}; //namespace mercury


#endif //__MERCURY_FLAT_SEGMENT_SEARCHER_H__
