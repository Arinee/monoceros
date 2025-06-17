/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     distance_refiner.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of DistanceRefiner
 */

#ifndef __MERCURY_ORIG_DIST_REFINER_H__
#define __MERCURY_ORIG_DIST_REFINER_H__

#include "framework/index_framework.h"
#include "index/base_index_provider.h"
#include "utils/my_heap.h"
#include "index/general_search_context.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

class DistanceRefiner
{
public:
    typedef std::shared_ptr<DistanceRefiner> Pointer;

    class Factory {
    public:
        Factory()
            :_featureProfiles(nullptr)
        {}
        ~Factory() 
        {
            if (!_featureProfiles) {
                delete[] _featureProfiles;
            }
        }
        int Init(BaseIndexProvider* indexProvider)
        {
            _indexMeta = *indexProvider->get_index_meta();
            auto segmentIndexes = indexProvider->get_segment_list();
            size_t segNum = segmentIndexes.size();
            _featureProfiles = new ArrayProfile*[segNum];
            for (size_t i = 0; i < segNum; ++i) {
                _featureProfiles[i] = segmentIndexes[i]->getFeatureProfile();
            }
            return 0;
        }
        DistanceRefiner Create(GeneralSearchContext *context)
        {
            DistanceRefiner r;
            r._featureProfiles = _featureProfiles;
            r._elemSize = _indexMeta.sizeofElement();
            r._measure = _indexMeta.measure();
            if (context->getSearchMethod() != IndexDistance::kMethodUnknown) {
                r._measure = IndexDistance::EmbodyMeasure(context->getSearchMethod());
            }
            return r;
        }
    private:
        IndexMeta _indexMeta;
        ArrayProfile** _featureProfiles;
        size_t _elemSize;
    };
public:

    void ScoreAndPush(const void *query, MyHeap<DistNode>& pqHeap, MyHeap<DistNode>& finalHeap)
    {
        for (auto &node: pqHeap.getData()) {
            docid_t docid = GET_DOCID(node.key);
            segid_t segid = GET_SEGID(node.key);
            const void* feature = _featureProfiles[segid]->getInfo(docid);
            float dist = _measure(query, feature, _elemSize);
            finalHeap.push(DistNode(node.key, dist));
        }
    }
private:
    size_t _elemSize;
    IndexDistance::Measure _measure;
    ArrayProfile** _featureProfiles;
};

} // namespace mercury

#endif // __MERCURY_ORIG_DIST_REFINER_H__
