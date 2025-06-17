/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     pq_dist_scorer.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of PqDistScorer
 */

#ifndef __MERCURY_PQ_DIST_SCORER_H__
#define __MERCURY_PQ_DIST_SCORER_H__

#include "framework/index_framework.h"
#include "query_distance_matrix.h"
#include "array_profile.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

class PqDistScorer 
{
public:
    typedef std::shared_ptr<PqDistScorer> Pointer;
public:
    PqDistScorer(ArrayProfile* profile, QueryDistanceMatrix* qdm) 
        : _pqCodeProfile(profile),
        _qdm(qdm)
    { }

    score_t score(docid_t docid)
    {
        const uint16_t *pqcode = 
            reinterpret_cast<const uint16_t *>(_pqCodeProfile->getInfo(docid));
        return innerScore(pqcode, _qdm);
    }

    // product quantizer calc
    static score_t innerScore(const uint16_t* codeFeature, QueryDistanceMatrix *qdm)
    {
        const size_t step = 8;
        size_t fragmentNum = qdm->getFragmentNum();
        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;
        float a4 = 0.0f;
        float a5 = 0.0f;
        float a6 = 0.0f;
        float a7 = 0.0f;
        size_t i = 0;
        score_t total = 0.0f;
        while (i + step < fragmentNum) {
            a0 = qdm->getDistance(codeFeature[i],   i );
            a1 = qdm->getDistance(codeFeature[i+1], i+1);
            a2 = qdm->getDistance(codeFeature[i+2], i+2);
            a3 = qdm->getDistance(codeFeature[i+3], i+3);
            a4 = qdm->getDistance(codeFeature[i+4], i+4);
            a5 = qdm->getDistance(codeFeature[i+5], i+5);
            a6 = qdm->getDistance(codeFeature[i+6], i+6);
            a7 = qdm->getDistance(codeFeature[i+7], i+7);
            i += step;
            total += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7; 
        }   
        for (; i < fragmentNum; i++) {
            total += qdm->getDistance(codeFeature[i], i); 
        }   
        return total;
    }

private:
    ArrayProfile* _pqCodeProfile;
    QueryDistanceMatrix* _qdm;
};

} // namespace mercury

#endif // __MERCURY_PQ_DIST_SCORER_H__
