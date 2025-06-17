/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    abstract structure for an Index
 */

#ifndef __MERCURY_INDEX_INDEX_H__
#define __MERCURY_INDEX_INDEX_H__

#include <cstdio>
#include <string>
#include "framework/index_meta.h"
#include "framework/index_params.h"
#include "framework/index_package.h"
#include "framework/index_logger.h"
#include "framework/index_distance.h"
#include "framework/index_framework.h"
#include "framework/utility/bitset_helper.h"
#include "common/common_define.h"
#include "utils/string_util.h"
#include "utils/hash_table.h"
#include "utils/deletemap.h"
#include "utils/dump_context.h"
#include "array_profile.h"
#include <memory>

namespace mercury {

class Index 
{
public:
    typedef std::shared_ptr<Index> Pointer;
    /// ctor and dtor
    Index (){
        _pPKProfile = new ArrayProfile;
        _pFeatureProfile = new ArrayProfile;
        _pIDMap=  new HashTable<uint64_t, docid_t>;
        _pDeleteMap = new mercury::BitsetHelper;
    }

    virtual ~Index (){
        DELETE_AND_SET_NULL(index_meta_);
        DELETE_AND_SET_NULL(index_params_);
        DELETE_AND_SET_NULL(_pPKProfile);
        DELETE_AND_SET_NULL(_pFeatureProfile);
        DELETE_AND_SET_NULL(_pIDMap);
        DELETE_AND_SET_NULL(_pDeleteMap);
    };

    /// create new seg, only use meta msg and index will save this handle
    virtual bool Create(IndexStorage::Pointer storage, 
            const std::string& file_name, IndexStorage::Handler::Pointer &&meta_file_handle) = 0;
    /// load index from index package file and index will save this handle
    virtual bool Load(IndexStorage::Handler::Pointer &&file_handle) = 0;
    /// unload index
    virtual void UnLoad() = 0;
    /// dump index to direction
    virtual bool Dump(IndexStorage::Pointer storage, const std::string& file_name, bool only_dump_meta) = 0;
    /// add a new vectoV
    virtual int Add(docid_t doc_id, uint64_t key, const void *val, size_t len) = 0;
    /// removes IDs from the index. Not supported by all indexes
    virtual bool RemoveId(uint64_t key) = 0;
    /// Display the actual class name and some more info
    virtual void Display() const{};
    /// whither index segment full
    virtual bool IsFull();
    /// return new empty index object
    virtual Index* CloneEmptyIndex() = 0;

    bool LoadIndexFromPackage(IndexPackage &package, bool only_dump_meta = false);
    bool DumpIndexToPackage(IndexPackage &package, bool only_dump_meta, DumpContext& dump_context);
    bool CreateIndexFromPackage(std::map<std::string, size_t>& stab);

    void set_index_meta(IndexMeta* index_meta);
    const IndexMeta* get_index_meta(void) const;

    void set_index_params(IndexParams* index_params);
    IndexParams* get_index_params();

    /// get value from profile
    uint64_t getPK(docid_t Doc_Id);
    const void *getFeature(docid_t Doc_Id);
    ArrayProfile* getPrimaryKeyProfile(void)
    {
        return _pPKProfile;
    }
    ArrayProfile* getFeatureProfile(void)
    {
        return _pFeatureProfile;
    }
    HashTable<uint64_t, docid_t> *getIdMap(void)
    {
        return _pIDMap;
    }

    /// profile ops
    int AddProfile(docid_t doc_id, uint64_t key, const void *val, size_t len);
    /// read index params
    void ReaIndexParams();

    /// Read doc num
    size_t get_doc_num(void) {
        return doc_num_;
    }

    const BitsetHelper *getDelMap() const {
        return _pDeleteMap;
    }

    bool DeleteMapEmpty(void) const {
        return _pDeleteMap->testNone();
    }

    const IndexMeta *getIndexMeta() const {
        return index_meta_;
    }

protected:
    /// index meta, must alloc new when create a index
    IndexMeta* index_meta_ = nullptr;
    /// index params, used when create new segment, must alloc new when create a index
    IndexParams* index_params_ = nullptr;
    /// index storage
    IndexStorage::Handler::Pointer stg_handler_;
    /// segment begin id
    size_t segment_id_begin_ = 0;
    /// doc_num
    size_t doc_num_ = 0;
    /// vector size
    size_t feature_info_size_ = 0;

    /// TODO remove it
    std::string buf_meta_data_;
public:
    // TODO: provide get function
    // docid -> busid 
    ArrayProfile *_pPKProfile = nullptr;
    // docid -> vector detail
    ArrayProfile *_pFeatureProfile = nullptr;
    // busid -> docid
    HashTable<uint64_t, docid_t> *_pIDMap = nullptr;
    // deletemap
    BitsetHelper *_pDeleteMap = nullptr;
};

} // namespace mercury

#endif // __MERCURY_INDEX_INDEX_H__
