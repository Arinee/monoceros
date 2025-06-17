/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-14 17:52

#include "index_searcher.h"
#include "redindex_iterator.h"
#include "src/core/framework/index_params.h"
#include "src/core/algorithm/general_search_context.h"

namespace mercury {
namespace redindex {

 int IndexSearcher::LoadDiskIndex(const std::string& disk_index_path, const std::string& medoids_data_path) {
    if (core_searcher_->LoadDiskIndex(disk_index_path, medoids_data_path) != 0) {
        std::cerr << "Failed to load index." << std::endl;
        return -1;
    }
    return 0;
 }

int IndexSearcher::LoadIndex(const std::string& path) {
    if (core_searcher_->LoadIndex(path) != 0) {
        std::cerr << "Failed to load index." << std::endl;
        return -1;
    }
    return 0;
}

int IndexSearcher::LoadIndex(const void* data, size_t size) {
    if (core_searcher_->LoadIndex(data, size) != 0) {
        std::cerr << "Failed to load index." << std::endl;
        return -1;
    }

    return 0;
}

void IndexSearcher::SetIndex(core::Index::Pointer index) {
    core_searcher_->SetIndex(index);
}

void IndexSearcher::SetBaseDocId(core::exdocid_t baseDocId) {
    core_searcher_->SetBaseDocId(baseDocId);
}

void IndexSearcher::SetDeletionMapRetriever(const core::DeletionMapRetriever& retriever) {
    core_searcher_->SetDeletionMapRetriever(retriever);
}

void IndexSearcher::SetVectorRetriever(const core::AttrRetriever& retriever) {
    core_searcher_->SetVectorRetriever(retriever);
}

mercury::core::Searcher::Pointer IndexSearcher::getCoreSearcher() {
    return core_searcher_;
}

void IndexSearcher::Search(const mercury::core::QueryInfo& queryInfo, mercury::core::GeneralSearchContext* context) const
{
    int ret = -1;
    if (context != nullptr) {
        ret = core_searcher_->Search(queryInfo, context);
        if (ret != 0) {
            std::cerr << "Failed to call index search." << std::endl;
            return;
        }

        return;
    }

    mercury::core::IndexParams index_params;
    mercury::core::GeneralSearchContext empty_context(index_params);
    ret = core_searcher_->Search(queryInfo, &empty_context);

    if (ret != 0) {
        std::cerr << "Failed to call index search." << std::endl;
        return;
    }

    return;
}

int64_t IndexSearcher::UsedMemoryInLoad() const {
    return 0xFFFF;
}

} // namespace redindex
} // namespace mercury
