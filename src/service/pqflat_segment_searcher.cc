/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     ivfpq_segment_searcher.cc
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    implement of mercury PqflatSegmentSearcher
 */

#include "pqflat_segment_searcher.h"
#include "index/pq_dist_scorer.h"
#include <sys/time.h>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace mercury {

void PqflatSegmentSearcher::seekAndPush(QueryDistanceMatrix *qdm, MyHeap<DistNode> &result)
{
    PqDistScorer scorer(pqCodeProfile, qdm);
    if (deleteFilter->empty()) {
        for (docid_t docid = 0; docid < docNum; ++docid) {
            float dist = scorer.score(docid);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    } else {
        for (docid_t docid = 0; docid < docNum; ++docid) {
            if (unlikely(deleteFilter->deleted(docid))) {
                continue;
            }
            float dist = scorer.score(docid);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    }
}

void PqflatSegmentSearcherWithFeeder::seekAndPush(QueryDistanceMatrix *qdm, MyHeap<DistNode> &result) 
{
    PqDistScorer scorer(pqCodeProfile, qdm);
    uint64_t key = INVALID_KEY;
    if (deleteFilter->empty()) {
        while ((key = (*customFeeder)())!= INVALID_KEY) {
            docid_t docid = INVALID_DOCID;
            if (unlikely(!idMap->find(key, docid))) {
                LOG_WARN("IdMap find key(%lu) error.", key);
                continue;
            }
            float dist = scorer.score(docid);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    } else {
        while ((key = (*customFeeder)())!= INVALID_KEY) {
            docid_t docid = INVALID_DOCID;
            if (unlikely(!idMap->find(key, docid))) {
                LOG_WARN("IdMap find key(%lu) error.", key);
                continue;
            }
            if (unlikely(deleteFilter->deleted(docid))) {
                continue;
            }
            float dist = scorer.score(docid);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    }
}

void PqflatSegmentSearcherWithFilter::seekAndPush(QueryDistanceMatrix *qdm, MyHeap<DistNode> &result) 
{
    PqDistScorer scorer(pqCodeProfile, qdm);
    if (deleteFilter->empty()) {
        for (docid_t docid = 0; docid < docNum; ++docid) {
            const key_t* keyPtr = (const key_t*)primaryKeyProfile->getInfo(docid);
            if (unlikely(keyPtr && (*customFilter)(*keyPtr))) {
                continue;
            }
            float dist = scorer.score(docid);
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
            float dist = scorer.score(docid);
            gloid_t gloid = GET_GLOID(segid, docid);
            result.push(DistNode(gloid, dist));
        }
    }
}

int PqflatSegSearcherFactory::Unload(void)
{
    _indexes.clear();
    _deleteFilters.clear();
    return 0;
}

int PqflatSegSearcherFactory::Cleanup(void)
{
    return 0;
}

int PqflatSegSearcherFactory::Init(const IndexParams& /*params*/)
{
    return 0;
}


int PqflatSegSearcherFactory::Load(BaseIndexProvider* indexProvider)
{
    if (!indexProvider) {
        return -1;
    }
    _indexes = indexProvider->get_segment_list();
    for (size_t segid = 0; segid < _indexes.size(); ++segid) {
        IndexPqflat* index = dynamic_cast<IndexPqflat*>(_indexes[segid].get());
        if (!index) {
            return -1;
        }
        LOG_INFO("Segment[%lu] type: pqflat, docNum: %lu", segid, index->get_doc_num());

        DeleteFilter::Pointer deleteFilter(new DeleteFilter(index));
        _deleteFilters.push_back(deleteFilter);
    }
    return 0;
}

PqflatSegmentSearcher::Pointer 
PqflatSegSearcherFactory::Make(size_t segid, const void* query, size_t bytes, GeneralSearchContext* context)
{
    const CustomFeeder &customFeeder = context->getFeeder();
    const CustomFilter &customFilter = context->getFilter();
    IndexPqflat *index = dynamic_cast<IndexPqflat*>(_indexes[segid].get());
    if (customFeeder.isValid()) {
        auto s = new PqflatSegmentSearcherWithFeeder(segid);
        s->docNum = index->get_doc_num();
        s->deleteFilter = _deleteFilters[segid].get();
        s->query = query;
        s->bytes = bytes;
        s->context = context;
        s->pqCodeProfile = index->getPqCodeProfile();
        s->idMap = index->getIdMap();
        s->customFeeder = &customFeeder;
        return PqflatSegmentSearcher::Pointer(s);
    }

    if (customFilter.isValid()) {
        auto s = new PqflatSegmentSearcherWithFilter(segid);
        s->docNum = index->get_doc_num();
        s->deleteFilter = _deleteFilters[segid].get();
        s->query = query;
        s->bytes = bytes;
        s->context = context;
        s->pqCodeProfile = index->getPqCodeProfile();
        s->customFilter = &customFilter;
        s->primaryKeyProfile = index->getPrimaryKeyProfile();
        return PqflatSegmentSearcher::Pointer(s);
    }

    auto s = new PqflatSegmentSearcher(segid);
    s->docNum = index->get_doc_num();
    s->deleteFilter = _deleteFilters[segid].get();
    s->query = query;
    s->bytes = bytes;
    s->context = context;
    s->pqCodeProfile = index->getPqCodeProfile();
    return PqflatSegmentSearcher::Pointer(s);
}

}; //namespace mercury
