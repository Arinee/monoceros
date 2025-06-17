/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_flat.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Index Flat
 */

#ifndef __MERCURY_INDEX_INDEX_FLAT_H__
#define __MERCURY_INDEX_INDEX_FLAT_H__

#include "index.h"
#include "general_search_context.h"

namespace mercury {

class IndexFlat : public Index
{
public:
    typedef std::shared_ptr<IndexFlat> Pointer;
    ///function interface
    IndexFlat () {
    }

    virtual ~IndexFlat (){
    }
    virtual void UnLoad() override 
    {};

    virtual bool Load(IndexStorage::Handler::Pointer &&file_handle) override;
    virtual bool Dump(IndexStorage::Pointer storage, const std::string& file_name, bool only_dump_meta) override;
    virtual bool Create(IndexStorage::Pointer storage, 
            const std::string& file_name, IndexStorage::Handler::Pointer &&meta_file_handle) override;
    virtual int Add(docid_t doc_id, uint64_t key, const void *val, size_t len) override;
    virtual bool RemoveId(uint64_t key) override;

    GENERATE_RETURN_EMPTY_INDEX(IndexFlat);
protected:
};

} // namespace mercury

#endif // __MERCURY_INDEX_INDEX_FLAT_H__
