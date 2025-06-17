/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     mult_cat_index.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     MAR 2019
 *   \version  1.0.0
 *   \brief    mult category index
 */

#ifndef _MERCURY_MULT_CAT_INDEX_H_
#define _MERCURY_MULT_CAT_INDEX_H_

#include <map>
#include "mult_slot_index.h"
#include "utils/hash_table.h"
#include "utils/dump_context.h"

namespace mercury {

class MultCatIndex
{
public:
    typedef std::shared_ptr<MultCatIndex> Pointer;

    MultCatIndex(IndexParams& index_params) : index_params_(index_params) {
        _coarseIndex = new MultSlotIndex<int64_t>();
        _IDMap = new HashTable<int, uint32_t>();
    }

    virtual ~MultCatIndex(){
        DELETE_AND_SET_NULL(_coarseIndex);
        DELETE_AND_SET_NULL(_IDMap);
    }

    // operator
    uint32_t GetCatNum();
    void MapSlot(int cateId, uint32_t soltId);

    /// load index from index package file and index will save this handle
    virtual bool Load(IndexStorage::Handler::Pointer &&file_handle);
    /// dump index to direction
    virtual bool Dump(IndexStorage::Pointer storage, const std::string& file_name);

    MultSlotIndex<int64_t> *GetMultIndex(){
        return _coarseIndex;
    }

    HashTable<int, uint32_t> *GetIDMap(){
        return _IDMap;
    }

    class CateFeeder{
    public:
        CateFeeder(MultSlotIndex<int64_t> *_coarseIndex , int slotId){
            if(_coarseIndex != nullptr){
                _posting = _coarseIndex->search(slotId);
            }
        }
        
        /// INVALID_DOCID
        int64_t GetNextDoc(){
            return _posting.next();
        }

        MultSlotIndex<int64_t>::PostingIterator _posting;
    };

    /// feeder
    CateFeeder GetCateFeeder(int cateId);
private:
    /// index params
    IndexParams index_params_;
    /// index storage
    IndexStorage::Handler::Pointer stg_handler_;
    /// coarse index 
    MultSlotIndex<int64_t> *_coarseIndex = nullptr;
    /// catid -> slot id
    HashTable<int, uint32_t> *_IDMap = nullptr; 
};

} // mercury

#endif //_MERCURY_MULT_CAT_INDEX_H_
