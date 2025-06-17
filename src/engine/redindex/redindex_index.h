/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: qiuming <qiuming@xiaohongshu.com>
/// Created: 2019-08-27 12:13

#pragma once

#include "redindex_common.h"
#include "src/core/algorithm/index.h"

namespace mercury { 
namespace redindex {

class RedIndex {
public:
    using Pointer = std::shared_ptr<RedIndex>;
    //TODO why? use gflags
    RedIndex(mercury::core::Index::Pointer core_index)
        : core_index_(core_index){}

    int Load(const void* data, size_t size) {
        return core_index_->Load(data, size);
    }

    int InitForLoad(const SchemaParams& schema) {
        mercury::core::IndexParams index_params;
        SchemaToIndexParam(schema, index_params);
        core_index_->SetIndexParams(index_params);
        return 0;
    }

    int Add(RedIndexDocid redindex_docid, const std::string& data_str) {
        return core_index_->Add(redindex_docid, 0, data_str);
    }

    size_t UsedMemoryInCurrent() const {
        return core_index_->UsedMemoryInCurrent();
    }
    int PrintStats() const;
    float GetRankScore(RedIndexDocid redindex_docid) const {
        return core_index_->GetRankScore(redindex_docid);
    }

    size_t get_current_doc_num() const {
        return (size_t)core_index_->GetDocNum();
    }

    mercury::core::Index::Pointer GetCoreIndex() const {
        return core_index_;
    }

private:
    RedIndex(const RedIndex &) = delete;
    RedIndex & operator =(RedIndex &) = delete;

private:
    mercury::core::Index::Pointer core_index_;
};

} // namespace redindex
} // namespace mercury
