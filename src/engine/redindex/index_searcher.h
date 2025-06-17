/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-14 17:44

#pragma once

#include "redindex_index.h"
#include "redindex_iterator.h"
#include "src/core/algorithm/searcher.h"

namespace mercury { 
namespace redindex {

class IndexSearcher {
public:
    using Pointer = std::shared_ptr<IndexSearcher>;
    IndexSearcher(core::Searcher::Pointer core_searcher)
        : core_searcher_(core_searcher){
    }
    virtual int LoadDiskIndex(const std::string& disk_index_path, const std::string& medoids_data_path);
    virtual int LoadIndex(const std::string& path);
    virtual int LoadIndex(const void* data, size_t size);
    virtual void SetIndex(core::Index::Pointer index);
    virtual void SetBaseDocId(core::exdocid_t baseDocId);
    virtual void SetDeletionMapRetriever(const core::DeletionMapRetriever& retriever);
    virtual void SetVectorRetriever(const core::AttrRetriever& retriever);
    virtual void Search(const mercury::core::QueryInfo& queryInfo,
                        mercury::core::GeneralSearchContext* context = nullptr) const;
    virtual int64_t UsedMemoryInLoad() const;
    mercury::core::Searcher::Pointer getCoreSearcher();

private:
    mercury::core::Searcher::Pointer core_searcher_;
};

} // namespace redindex
} // namespace mercury
