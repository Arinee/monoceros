#pragma once

#include "src/core/common/common.h"
#include "src/core/algorithm/coarse_index.h"

MERCURY_NAMESPACE_BEGIN(core);

class FlatCoarseIndex
{
public:
    FlatCoarseIndex()
        : slot_doc_ids_(nullptr), slot_doc_nums_(nullptr),
          slot_start_indexs_(nullptr){};

    ~FlatCoarseIndex()
    {
        if (slot_doc_ids_ != nullptr) {
            delete[] slot_doc_ids_;
            slot_doc_ids_ = nullptr;
        }
        if (slot_doc_nums_ != nullptr) {
            delete[] slot_doc_nums_;
            slot_doc_nums_ = nullptr;
        }
        if (slot_start_indexs_ != nullptr) {
            delete[] slot_start_indexs_;
            slot_start_indexs_ = nullptr;
        }
    }

    template <typename Block>
    void ConvertFromCoarseIndex(CoarseIndex<Block> &coarse_index)
    {
        uint32_t doc_num = coarse_index.getUsedDocNum();
        uint32_t slot_num = coarse_index.getSlotNum();

        slot_doc_ids_ = new uint32_t[doc_num];
        slot_doc_nums_ = new uint32_t[slot_num];
        slot_start_indexs_ = new uint32_t[slot_num];

        uint32_t curren_doc_index = 0;
        for (size_t i = 0; i < slot_num; i++) {
            auto iter = coarse_index.search(i);
            slot_doc_nums_[i] = iter.getDocNum();
            slot_start_indexs_[i] = curren_doc_index;
            while (!iter.finish()) {
                uint32_t docid = iter.next();
                slot_doc_ids_[curren_doc_index++] = docid;
            }
        }
    }

    uint32_t *GetSlotDocIds()
    {
        return slot_doc_ids_;
    }

    uint32_t *GetSlotDocNums()
    {
        return slot_doc_nums_;
    }

    uint32_t *GetStartIndexs()
    {
        return slot_start_indexs_;
    }

    uint32_t GetSlotDocId(size_t slot_index)
    {
        return slot_doc_ids_[slot_index];
    }

    uint32_t GetSlotDocNum(size_t slot_index)
    {
        return slot_doc_nums_[slot_index];
    }

    uint32_t GetStartIndexs(size_t slot_index)
    {
        return slot_start_indexs_[slot_index];
    }

private:
    uint32_t *slot_doc_ids_;
    uint32_t *slot_doc_nums_;
    uint32_t *slot_start_indexs_;
};

MERCURY_NAMESPACE_END(core);
