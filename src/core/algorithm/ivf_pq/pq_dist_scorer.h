/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: qiuming <qiuming@xiaohongshu.com>
/// Created: 2019-09-03 18:11

#pragma once

#include "query_distance_matrix.h"
#include "src/core/utils/array_profile.h"
#include <memory>
#include <string>
#include <vector>

MERCURY_NAMESPACE_BEGIN(core);

class PqDistScorer 
{
public:
    typedef std::shared_ptr<PqDistScorer> Pointer;
public:
    PqDistScorer(const ArrayProfile* profile)
        : _pqCodeProfile(profile)
    { }

    score_t score(docid_t docid, QueryDistanceMatrix* qdm)
    {
        const uint16_t *pqcode = 
            reinterpret_cast<const uint16_t *>(_pqCodeProfile->getInfo(docid));
        return innerScore(pqcode, qdm);
    }

    // product quantizer calc
    static score_t innerScore(const uint16_t* codeFeature, QueryDistanceMatrix* qdm)
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
    const ArrayProfile* _pqCodeProfile;
};

MERCURY_NAMESPACE_END(core);
