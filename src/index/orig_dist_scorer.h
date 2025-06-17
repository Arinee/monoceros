/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     orig_dist_scorer.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of OrigDistScorer
 */

#ifndef __MERCURY_ORIG_DIST_SCORER_H__
#define __MERCURY_ORIG_DIST_SCORER_H__

#include "framework/index_framework.h"
#include "index.h"
#include "general_search_context.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

class OrigDistScorer
{
public:
    typedef std::shared_ptr<OrigDistScorer> Pointer;

    class Factory 
    {
    public:
        Factory()
            :_featureProfile(nullptr)
        {}
        int Init(Index* index) 
        {
            _indexMeta = *index->get_index_meta();
            _featureProfile = index->getFeatureProfile();
            return 0;
        }
        OrigDistScorer Create(GeneralSearchContext* context)
        {
            OrigDistScorer s;
            s._elemSize = _indexMeta.sizeofElement();
            s._measure = _indexMeta.measure();
            if (context->getSearchMethod() != IndexDistance::kMethodUnknown) {
                s._measure = IndexDistance::EmbodyMeasure(context->getSearchMethod());
            }
            s._featureProfile = _featureProfile;
            return s;
        }
    private:
        IndexMeta _indexMeta;
        ArrayProfile* _featureProfile;
    };

public:
    OrigDistScorer()
        :_elemSize(0),
        _featureProfile(nullptr)
    {}

    score_t Score(docid_t docid, const void* query)
    {
        const void* feature = _featureProfile->getInfo(docid);
        return _measure(query, feature, _elemSize);
    }
private:
    size_t _elemSize;
    IndexDistance::Measure _measure;
    ArrayProfile* _featureProfile;
};

} // namespace mercury

#endif // __MERCURY_ORIG_DIST_SCORER_H__
