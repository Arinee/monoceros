/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     ivfpq_segment_searcher.cc
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    implement of mercury IvfpqSegmentSearcher
 */

#include "ivfpq_segment_searcher.h"
#include <sys/time.h>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace mercury {

void IvfpqSegmentSearcher::seekAndPush(QueryDistanceMatrix *qdm, MyHeap<DistNode> &result)
{
    PqDistScorer scorer(pqCodeProfile, qdm);
    std::vector<CoarseIndex::PostingIterator> &&
        postings = ivfSeeker->Seek(query, bytes, context);
    if (deleteFilter->empty()) {
        for (auto &posting : postings) {
            docid_t docid = INVALID_DOCID;
            while ((docid = posting.next()) != INVALID_DOCID) {
                float dist = scorer.score(docid);
                gloid_t gloid = GET_GLOID(segid, docid);
                result.push(DistNode(gloid, dist));
            }
        }
    } else {
        for (auto &posting : postings) {
            docid_t docid = INVALID_DOCID;
            while ((docid = posting.next()) != INVALID_DOCID) {
                if (unlikely(deleteFilter->deleted(docid))) {
                    continue;
                }
                float dist = scorer.score(docid);
                gloid_t gloid = GET_GLOID(segid, docid);
                result.push(DistNode(gloid, dist));
            }
        }
    }
}

void IvfpqSegmentSearcherWithFilter::seekAndPush(QueryDistanceMatrix *qdm, MyHeap<DistNode> &result)
{
    PqDistScorer scorer(pqCodeProfile, qdm);
    std::vector<CoarseIndex::PostingIterator> &&
        postings = ivfSeeker->Seek(query, bytes, context);
    if (deleteFilter->empty()) {
        for (auto &posting : postings) {
            docid_t docid = INVALID_DOCID;
            while ((docid = posting.next()) != INVALID_DOCID) {
                const key_t *keyPtr = (const key_t *)primaryKeyProfile->getInfo(docid);
                if (unlikely(keyPtr && (*customFilter)(*keyPtr))) {
                    continue;
                }
                float dist = scorer.score(docid);
                gloid_t gloid = GET_GLOID(segid, docid);
                result.push(DistNode(gloid, dist));
            }
        }
    } else {
        for (auto &posting : postings) {
            docid_t docid = INVALID_DOCID;
            while ((docid = posting.next()) != INVALID_DOCID) {
                if (unlikely(deleteFilter->deleted(docid))) {
                    continue;
                }
                const key_t *keyPtr = (const key_t *)primaryKeyProfile->getInfo(docid);
                if (unlikely(keyPtr && (*customFilter)(*keyPtr))) {
                    continue;
                }
                float dist = scorer.score(docid);
                gloid_t gloid = GET_GLOID(segid, docid);
                result.push(DistNode(gloid, dist));
            }
        }
    }
}

void IvfpqSegmentSearcherWithFeeder::seekAndPush(QueryDistanceMatrix *qdm, MyHeap<DistNode> &result)
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

int IvfpqSegSearcherFactory::Unload(void)
{
    _indexes.clear();
    _ivfSeekers.clear();
    _deleteFilters.clear();
    return 0;
}

int IvfpqSegSearcherFactory::Cleanup(void)
{
    return 0;
}

int IvfpqSegSearcherFactory::Init(const IndexParams& params)
{
    std::string coarseScanRatioStr = params.getString(PARAM_PQ_SEARCHER_COARSE_SCAN_RATIO);
    std::vector<float> coarseScanRatioVec;
    StringUtil::fromString(coarseScanRatioStr, coarseScanRatioVec, ",");
    if (coarseScanRatioVec.size() == 0) {
        coarseScanRatioVec.push_back(LEVEL_SCAN_RATIO);
        coarseScanRatioVec.push_back(COARSE_PROBE_RATIO);
    }
    if (coarseScanRatioVec.size() == 1) {
        coarseScanRatioVec.insert(coarseScanRatioVec.begin(), LEVEL_SCAN_RATIO);
    }
    _defaultCoarseScanRatio = coarseScanRatioVec;
    LOG_INFO("Read default coarseScanRatio: %f, %f",
            _defaultCoarseScanRatio[0], _defaultCoarseScanRatio[1]);
    return 0;
}


int IvfpqSegSearcherFactory::Load(IvfpqIndexProvider* indexProvider)
{
    if (!indexProvider) {
        return -1;
    }
    _indexes = indexProvider->get_segment_list();
    for (size_t segid = 0; segid < _indexes.size(); ++segid) {
        IndexIvfpq* index = dynamic_cast<IndexIvfpq*>(_indexes[segid].get());
        if (!index) {
            return -1;
        }
        LOG_INFO("Segment[%lu] type: ivfpq, docNum: %lu", segid, index->get_doc_num());
        IvfSeeker::Pointer ivf_seeker(new IvfSeeker(_defaultCoarseScanRatio));
        if (ivf_seeker->Init(index) != 0) {
            LOG_ERROR("ivf seeker init error.");
            return -1;
        }
        _ivfSeekers.push_back(ivf_seeker);

        DeleteFilter::Pointer deleteFilter(new DeleteFilter(index));
        _deleteFilters.push_back(deleteFilter);
    }
    return 0;
}

IvfpqSegmentSearcher::Pointer 
IvfpqSegSearcherFactory::Make(size_t segid, const void* query, size_t bytes, GeneralSearchContext* context)
{
    IndexIvfpq *index = dynamic_cast<IndexIvfpq*>(_indexes[segid].get());
    const CustomFeeder &customFeeder = context->getFeeder();
    if (customFeeder.isValid()) {
        auto s = new IvfpqSegmentSearcherWithFeeder(segid);
        s->ivfSeeker = _ivfSeekers[segid].get();
        s->deleteFilter = _deleteFilters[segid].get();
        s->query = query;
        s->bytes = bytes;
        s->context = context;
        s->pqCodeProfile= index->getPqCodeProfile();
        s->idMap = index->getIdMap();
        s->customFeeder = &customFeeder;
        return IvfpqSegmentSearcher::Pointer(s);
    }

    const CustomFilter &customFilter = context->getFilter();
    if (customFilter.isValid()) {
        auto s = new IvfpqSegmentSearcherWithFilter(segid);
        s->ivfSeeker = _ivfSeekers[segid].get();
        s->deleteFilter = _deleteFilters[segid].get();
        s->query = query;
        s->bytes = bytes;
        s->context = context;
        s->pqCodeProfile= index->getPqCodeProfile();
        s->primaryKeyProfile = index->getPrimaryKeyProfile();
        s->customFilter = &customFilter;
        return IvfpqSegmentSearcher::Pointer(s);
    }

    auto s = new IvfpqSegmentSearcher(segid);
    s->ivfSeeker = _ivfSeekers[segid].get();
    s->deleteFilter = _deleteFilters[segid].get();
    s->query = query;
    s->bytes = bytes;
    s->context = context;
    s->pqCodeProfile= index->getPqCodeProfile();
    return IvfpqSegmentSearcher::Pointer(s);
}

}; //namespace mercury
