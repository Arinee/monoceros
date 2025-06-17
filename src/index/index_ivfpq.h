/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_ivfpq.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Index Ivfpq
 */

#ifndef __MERCURY_INDEX_INDEX_IVFPQ_H__
#define __MERCURY_INDEX_INDEX_IVFPQ_H__

#include "index_ivfflat.h"
#include "query_distance_matrix.h"
#include "general_search_context.h"

namespace mercury {

class IndexIvfpq : public IndexIvfflat
{
public:
    typedef std::shared_ptr<IndexIvfpq> Pointer;
    ///function interface
    IndexIvfpq (){
        _pqcodeProfile = new ArrayProfile;
    }

    ~IndexIvfpq () override {
        DELETE_AND_SET_NULL(_pqcodeProfile);
    }

    virtual bool Load(IndexStorage::Handler::Pointer &&file_handle) override;
    virtual bool Dump(IndexStorage::Pointer storage, const std::string& file_name, bool only_dump_meta) override;
    virtual bool Create(IndexStorage::Pointer storage, 
            const std::string& file_name, 
            IndexStorage::Handler::Pointer &&meta_file_handle) override;
    virtual int Add(docid_t doc_id, uint64_t key, const void *val, size_t len) override;

    /// load PQ index data
    bool CreatePQIndexFromPackage(std::map<std::string, size_t>& stab);
    bool LoadPQIndexFromPackage(IndexPackage &package);
    bool DumpPQIndexToPackage(IndexPackage &package, bool only_dump_meta, DumpContext& dump_context);        

    void ReadPQParams();
    QueryDistanceMatrix::Pointer InitQueryDistanceMatrix(const void *query, GeneralSearchContext *context);
    std::vector<size_t> computeRoughLevelScanLimit(GeneralSearchContext* pq_context);

    //TODO must init qdm before
    inline float CalcDistance(docid_t doc_id, QueryDistanceMatrix *qdm)
    {   
        float score = std::numeric_limits<float>::max();
        const uint16_t *pqcode = reinterpret_cast<const uint16_t *>(_pqcodeProfile->getInfo(doc_id));
        score = calcScore(pqcode, qdm);

        return score;
    }

    inline const uint16_t * getPqCode(docid_t doc_id)
    {   
        return reinterpret_cast<const uint16_t *>(_pqcodeProfile->getInfo(doc_id));
    }

    inline ArrayProfile* getPqCodeProfile(void) 
    {   
        return _pqcodeProfile;
    }


    // product quantizer calc
    static score_t calcScore(const uint16_t* codeFeature, QueryDistanceMatrix *qdm)
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

    GENERATE_RETURN_EMPTY_INDEX(IndexIvfpq);
    // docid -> code
    ArrayProfile *_pqcodeProfile;
protected:
    size_t product_info_size_ = 0;
};
} // namespace mercury

#endif // __MERCURY_INDEX_INDEX_IVFPQ_H__
