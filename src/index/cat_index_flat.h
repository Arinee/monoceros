/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cat_index_flat.h
 *   \author   jiazi@xiaohongshu.com
 *   \date     Apr 2019
 *   \version  1.0.0
 *   \brief    Cat Index Flat
 */

#ifndef __MERCURY_CAT_INDEX_INDEX_FLAT_H__
#define __MERCURY_CAT_INDEX_INDEX_FLAT_H__

#include "index_flat.h"
#include "coarse_index.h"

namespace mercury {

class CatIndexFlat : public IndexFlat
{
public:
    typedef std::shared_ptr<CatIndexFlat> Pointer;
    ///function interface
    CatIndexFlat () : _catSlotMap(new HashTable<cat_t, slot_t>()),
                    _slotDocIndex(new CoarseIndex()) {}

    ~CatIndexFlat() = default; 

    void UnLoad() override 
    {}

    bool Load(IndexStorage::Handler::Pointer &&file_handle) override;
    bool Dump(IndexStorage::Pointer storage, const std::string& file_name, bool only_dump_meta) override;
    bool Create(IndexStorage::Pointer storage, 
            const std::string& file_name, IndexStorage::Handler::Pointer &&meta_file_handle) override;
    int Add(docid_t doc_id, uint64_t key, const void *val, size_t len) override;
    bool RemoveId(uint64_t key) override;

    CoarseIndex::PostingIterator GetPostingIter(const cat_t cat_) const {
        slot_t slot = 0;
        if (!_catSlotMap->find(cat_, slot)) {
            //LOG_ERROR("failed to find slot id for cat: %lu", cat_);
            return CoarseIndex::PostingIterator();
        }
        return _slotDocIndex->search(slot);
    }

    GENERATE_RETURN_EMPTY_INDEX(CatIndexFlat);
protected:
    std::shared_ptr<HashTable<cat_t, slot_t>> _catSlotMap = nullptr;
    std::shared_ptr<CoarseIndex> _slotDocIndex = nullptr;
private:
    using Base = IndexFlat;
};

} // namespace mercury

#endif // __MERCURY_CAT_INDEX_FLAT_H__
