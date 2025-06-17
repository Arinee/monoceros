/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_ivfflat.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Index Ivfflat
 */

#ifndef __MERCURY_INDEX_INDEX_IVFFLAT_H__
#define __MERCURY_INDEX_INDEX_IVFFLAT_H__

#include "index_ivf.h"
#include "general_search_context.h"
#include "ivf_posting_iterator.h"

namespace mercury {

class IndexIvfflat : public IndexIvf
{
public:
    typedef std::shared_ptr<IndexIvfflat> Pointer;
    ///function interface
    IndexIvfflat () {
        coarse_index_ = new CoarseIndex;
    }

    virtual ~IndexIvfflat (){
        DELETE_AND_SET_NULL(coarse_index_);
    }

    virtual void UnLoad() override 
    {};

    CoarseIndex *get_coarse_index()
    {
        return coarse_index_;
    }
    
    virtual bool Load(IndexStorage::Handler::Pointer &&file_handle) override;
    virtual bool Dump(IndexStorage::Pointer storage, const std::string& file_name, bool only_dump_meta) override;
    virtual bool Create(IndexStorage::Pointer storage, 
            const std::string& file_name, IndexStorage::Handler::Pointer &&meta_file_handle) override;
    virtual int Add(docid_t doc_id, uint64_t key, const void *val, size_t len) override;
    virtual bool RemoveId(uint64_t key) override;

    /// calc pre doc
    float CalcDistance(const void *query, size_t len, docid_t doc_id);

    /// load flat index data
    bool LoadFLatIndexFromPackage(IndexPackage &package);
    bool DumpFLatIndexToPackage(IndexPackage &package, bool only_dump_meta, DumpContext& dump_context);
    bool CreateFLatIndexFromPackage(std::map<std::string, size_t>& stab);

    GENERATE_RETURN_EMPTY_INDEX(IndexIvfflat);
public:
    /// corase index posting
    CoarseIndex *coarse_index_ = nullptr;
    /// coarse centroid _slotNum
    size_t slot_num_ = 0;
};

inline float IndexIvfflat::CalcDistance(const void *query, size_t /*len*/, docid_t doc_id)
{   
    float score = std::numeric_limits<float>::max();
    const void *val = getFeature(doc_id);
    score = index_meta_->distance(val, query);
    return score;
}

} // namespace mercury

#endif // __MERCURY_INDEX_INDEX_IVFFLAT_H__
