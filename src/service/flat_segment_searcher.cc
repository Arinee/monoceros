/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     flat_segment_searcher.cc
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    implement of mercury FlatSegmentSearcher
 */

#include "flat_segment_searcher.h"
#include <sys/time.h>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace mercury {

void FlatSegmentSearcher::seekAndPush(MyHeap<DistNode> &result)
{
    if (deleteFilter->empty()) {
        for (docid_t docid = 0; docid < docNum; ++docid) {
            float dist = scorer.Score(docid, query);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    } else {
        for (docid_t docid = 0; docid < docNum; ++docid) {
            if (unlikely(deleteFilter->deleted(docid))) {
                continue;
            }
            float dist = scorer.Score(docid, query);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    }
}

void FlatSegmentSearcherWithFeeder::seekAndPush(MyHeap<DistNode> &result) 
{
    if (deleteFilter->empty()) {
        uint64_t key = INVALID_KEY;
        while ((key = (*customFeeder)())!= INVALID_KEY) {
            docid_t docid = INVALID_DOCID;
            if (unlikely(!idMap->find(key, docid))) {
                LOG_WARN("IdMap find key(%lu) error.", key);
                continue;
            }
            float dist = scorer.Score(docid, query);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    } else {
        uint64_t key = INVALID_KEY;
        while ((key = (*customFeeder)())!= INVALID_KEY) {
            docid_t docid = INVALID_DOCID;
            if (unlikely(!idMap->find(key, docid))) {
                LOG_WARN("IdMap find key(%lu) error.", key);
                continue;
            }
            if (unlikely(deleteFilter->deleted(docid))) {
                continue;
            }
            float dist = scorer.Score(docid, query);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    }
}

void FlatSegmentSearcherWithDocFeeder::seekAndPush(MyHeap<DistNode> &result) 
{
    size_t visited = 0;
    if (deleteFilter->empty()) {
        docid_t docid = INVALID_DOCID;
        while ((docid = (*customDocFeeder)())!= INVALID_DOCID) {
            ++visited;
            float dist = scorer.Score(docid, query);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    } else {
        docid_t docid = INVALID_DOCID;
        while ((docid = (*customDocFeeder)())!= INVALID_DOCID) {
            if (unlikely(deleteFilter->deleted(docid))) {
                continue;
            }
            ++visited;
            float dist = scorer.Score(docid, query);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    }
    context->getVisitedDocNum() += visited;
}

void FlatSegmentSearcherWithFilter::seekAndPush(MyHeap<DistNode> &result) 
{
    if (deleteFilter->empty()) {
        for (docid_t docid = 0; docid < docNum; ++docid) {
            const key_t* keyPtr = (const key_t*)primaryKeyProfile->getInfo(docid);
            if (unlikely(keyPtr && (*customFilter)(*keyPtr))) {
                continue;
            }
            float dist = scorer.Score(docid, query);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    } else {
        for (docid_t docid = 0; docid < docNum; ++docid) {
            if (unlikely(deleteFilter->deleted(docid))) {
                continue;
            }
            const key_t* keyPtr = (const key_t*)primaryKeyProfile->getInfo(docid);
            if (unlikely(keyPtr && (*customFilter)(*keyPtr))) {
                continue;
            }
            float dist = scorer.Score(docid, query);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    }
}

int FlatSegSearcherFactory::Unload(void)
{
    _indexes.clear();
    _deleteFilters.clear();
    return 0;
}

int FlatSegSearcherFactory::Cleanup(void)
{
    return 0;
}

int FlatSegSearcherFactory::Init(const IndexParams& /*params*/)
{
    return 0;
}


int FlatSegSearcherFactory::Load(BaseIndexProvider* indexProvider)
{
    if (!indexProvider) {
        return -1;
    }
    _indexes = indexProvider->get_segment_list();
    for (size_t segid = 0; segid < _indexes.size(); ++segid) {
        Index *index = dynamic_cast<Index *>(_indexes[segid].get());
        if (!index) {
            return -1;
        }
        LOG_INFO("Segment[%lu] type: flat, docNum: %lu", segid, index->get_doc_num());

        OrigDistScorer::Factory scorerFactory;
        if (scorerFactory.Init(index) != 0) {
            LOG_ERROR("OrigDistScorer::Factory Init error.");
            return -1;
        }
        _scorerFactories.push_back(scorerFactory);

        DeleteFilter::Pointer deleteFilter(new DeleteFilter(index));
        _deleteFilters.push_back(deleteFilter);
    }
    return 0;
}

FlatSegmentSearcher::Pointer 
FlatSegSearcherFactory::Make(size_t segid, const void* query, size_t /*bytes*/, GeneralSearchContext* context)
{
    Index *index = dynamic_cast<Index *>(_indexes[segid].get());

    const CustomFeeder &customFeeder = context->getFeeder();
    if (customFeeder.isValid()) {
        auto s = new FlatSegmentSearcherWithFeeder(segid);
        s->deleteFilter = _deleteFilters[segid].get();
        s->query = query;
        s->context = context;
        s->idMap = index->getIdMap();
        s->customFeeder = &customFeeder;
        s->scorer = _scorerFactories[segid].Create(context);
        s->docNum = index->get_doc_num();
        return FlatSegmentSearcher::Pointer(s);
    }

    const CustomDocFeeder &customDocFeeder = context->getDocFeeder();
    if (customDocFeeder.isValid()) {
        auto s = new FlatSegmentSearcherWithDocFeeder(segid);
        s->deleteFilter = _deleteFilters[segid].get();
        s->query = query;
        s->context = context;
        s->idMap = index->getIdMap();
        s->customDocFeeder = &customDocFeeder;
        s->scorer = _scorerFactories[segid].Create(context);
        s->docNum = index->get_doc_num();
        return FlatSegmentSearcher::Pointer(s);
    }

    const CustomFilter &customFilter = context->getFilter();
    if (customFilter.isValid()) {
        auto s = new FlatSegmentSearcherWithFilter(segid);
        s->deleteFilter = _deleteFilters[segid].get();
        s->query = query;
        s->context = context;
        s->scorer = _scorerFactories[segid].Create(context);
        s->docNum = index->get_doc_num();
        s->customFilter = &customFilter;
        s->primaryKeyProfile = index->getPrimaryKeyProfile();
        return FlatSegmentSearcher::Pointer(s);
    }

    auto s = new FlatSegmentSearcher(segid);
    s->deleteFilter = _deleteFilters[segid].get();
    s->query = query;
    s->context = context;
    s->scorer = _scorerFactories[segid].Create(context);
    s->docNum = index->get_doc_num();
    return FlatSegmentSearcher::Pointer(s);
}

}; //namespace mercury
