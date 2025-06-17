#include "group_hnsw_merger.h"

MERCURY_NAMESPACE_BEGIN(core);

//! Initialize Merger
int GroupHnswMerger::Init(const IndexParams &params) {
    merged_index_ = nullptr;
    return 0;
}

int GroupHnswMerger::MergeByDocid(const std::vector<Index::Pointer> &indexes, const MergeIdMap& id_map) {
    if (indexes.size() != 1) {
        LOG_ERROR("HNSW build should be single thread! index size is: %zd", indexes.size());
        return -1;
    }
    merged_index_ = dynamic_cast<GroupHnswIndex*>(indexes.at(0).get());
    return 0;
}

int GroupHnswMerger::MergeByPk(const std::vector<Index::Pointer> &indexes, const MergePkMap& pk_map) {
    if (indexes.size() != 1) {
        LOG_ERROR("HNSW build should be single thread! index size is: %zd", indexes.size());
        return -1;
    }
    merged_index_ = dynamic_cast<GroupHnswIndex*>(indexes.at(0).get());
    return 0;
}

//! Dump index into file or memory
int GroupHnswMerger::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) {
    return -1;
}

const void * GroupHnswMerger::DumpIndex(size_t *size) {
    return merged_index_->DumpInMerger(*size);
}

MERCURY_NAMESPACE_END(core);
