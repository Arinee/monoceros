/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: qiuming <qiuming@xiaohongshu.com>
/// Created: 2019-08-30 11:54

#pragma once

#include <string>
#include <sys/types.h>
#include <vector>
#include <memory>
#include <pthread.h>
#include <utility>
#include "redindex_common.h"
#include "src/core/framework/search_result.h"

namespace mercury { 
namespace redindex {
// use example:
// RedIndexIterator::Pointer iter = Search();
// if (!iter) {
//     error_process()
// }
// for (iter->Init(); iter->Isvalid(); iter->Next()) {
//    auto d = iter->Data();
// }

class RedIndexIterator
{
public:
    typedef std::unique_ptr<RedIndexIterator> Pointer;

    virtual ~RedIndexIterator() = default;
    RedIndexIterator(const std::vector<mercury::core::SearchResult>&& search_results)
        : search_results_(std::move(search_results)), current_(0){
    }

    virtual void Init(){
        current_ = 0;
    };

    virtual void Next() {
        current_++;
    }

    virtual bool IsValid() const {
        return current_ < search_results_.size();
    }

    virtual RedIndexDocid Data() const {
        if (!IsValid()) {
            return INVALID_REDINDEX_DOC_ID;
        }

        return search_results_[current_].gloid;
    }

    virtual float Score() const {
        if (!IsValid()) {
            return INVALID_REDINDEX_DOC_ID;
        }

        return search_results_[current_].score;
    }

    size_t Size() {
        return search_results_.size();
    }

private:
    const std::vector<mercury::core::SearchResult> search_results_;
    mercury::core::docid_t current_;
};

} // name space redindex
} // name space mercury
