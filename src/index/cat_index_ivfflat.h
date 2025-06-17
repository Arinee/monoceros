/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cat_index_ivfflat.h
 *   \author   jiazi@xiaohongshu.com
 *   \date     Apr 2019
 *   \version  1.0.0
 *   \brief    Cat Index Ivfflat
 */

#ifndef __MERCURY_CAT_INDEX_IVFFLAT_H__
#define __MERCURY_CAT_INDEX_IVFFLAT_H__

#include "index_ivfflat.h"

namespace mercury {

class CatIndexIvfflat : public IndexIvfflat
{
public:
    typedef std::shared_ptr<CatIndexIvfflat> Pointer;
    CatIndexIvfflat ()
    {
        _keyCatMap = std::make_shared<HashTable<key_t, cat_t, 1>>();
        _catSet = std::make_shared<HashTable<cat_t, cat_t>>();
    }

    virtual ~CatIndexIvfflat () = default;

    std::shared_ptr<HashTable<key_t, cat_t, 1>> GetKeyCatMap() const {
        return _keyCatMap;
    }

    std::shared_ptr<HashTable<cat_t, cat_t>> GetCatSet() const {
        return _catSet;
    }

    bool Load(IndexStorage::Handler::Pointer &&file_handle) override;
    bool Dump(IndexStorage::Pointer storage, const std::string& file_name, bool only_dump_meta) override;
    bool Create(IndexStorage::Pointer storage, 
            const std::string& file_name, IndexStorage::Handler::Pointer &&meta_file_handle) override;
    int Add(docid_t doc_id, uint64_t key, const void *val, size_t len) override;
    bool RemoveId(uint64_t key) override;

    GENERATE_RETURN_EMPTY_INDEX(CatIndexIvfflat);
protected:
    std::shared_ptr<HashTable<key_t, cat_t, 1>> _keyCatMap;
    std::shared_ptr<HashTable<cat_t, cat_t>> _catSet;
};

} // namespace mercury

#endif // __MERCURY_CAT_INDEX_IVFFLAT_H__
