/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: qiuming <qiuming@xiaohongshu.com>
/// Created: 2019-09-03 18:11

#pragma once

#include "query_distance_matrix1.h"
#include "src/core/utils/array_profile.h"
#include <memory>
#include <string>
#include <vector>

MERCURY_NAMESPACE_BEGIN(core);

class PqDistScorer1 
{
public:
    typedef std::shared_ptr<PqDistScorer1> Pointer;
public:
    PqDistScorer1(const ArrayProfile* profile)
        : _pqCodeProfile(profile)
    {}

    score_t score(docid_t docid, QueryDistanceMatrix1* qdm, const distance_t* ipCentArray)
    {
        const uint8_t *pqcode = 
            reinterpret_cast<const uint8_t *>(_pqCodeProfile->getInfo(docid));
        return innerScore(pqcode, qdm, ipCentArray);
    }

    // product quantizer calc
    static score_t innerScore(const uint8_t* codeFeature, QueryDistanceMatrix1* qdm, const distance_t* ipCentArray)
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
            size_t index = i * qdm->getCentroidNum() + codeFeature[i];
            a0 = qdm->getIpValByIndex(index) + ipCentArray[index];
            index = (i + 1) * qdm->getCentroidNum() + codeFeature[i + 1];
            a1 = qdm->getIpValByIndex(index) + ipCentArray[index];
            index = (i + 2) * qdm->getCentroidNum() + codeFeature[i + 2];
            a2 = qdm->getIpValByIndex(index) + ipCentArray[index];
            index = (i + 3) * qdm->getCentroidNum() + codeFeature[i + 3];
            a3 = qdm->getIpValByIndex(index) + ipCentArray[index];
            index = (i + 4) * qdm->getCentroidNum() + codeFeature[i + 4];
            a4 = qdm->getIpValByIndex(index) + ipCentArray[index];
            index = (i + 5) * qdm->getCentroidNum() + codeFeature[i + 5];
            a5 = qdm->getIpValByIndex(index) + ipCentArray[index];
            index = (i + 6) * qdm->getCentroidNum() + codeFeature[i + 6];
            a6 = qdm->getIpValByIndex(index) + ipCentArray[index];
            index = (i + 7) * qdm->getCentroidNum() + codeFeature[i + 7];
            a7 = qdm->getIpValByIndex(index) + ipCentArray[index];
            i += step;
            total += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
        }
        for (; i < fragmentNum; i++) {
            size_t index = i * qdm->getCentroidNum() + codeFeature[i];
            total += qdm->getIpValByIndex(index) + ipCentArray[index];
        }
        return total;
    }

private:
    const ArrayProfile* _pqCodeProfile;
};

MERCURY_NAMESPACE_END(core);
