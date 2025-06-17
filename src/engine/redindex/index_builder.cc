/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-13 15:00

#include "index_builder.h"

namespace mercury {
namespace redindex {

int IndexBuilder::PreAdd(const std::string& build_str, float *score)
{
    return core_builder_->GetRankScore(build_str, score);
}

int IndexBuilder::Add(RedIndexDocid redindex_docid,
                      const std::string& query_str,
                      const std::string& primary_key) {
    if (core_builder_->AddDoc(redindex_docid, mercury::core::INVALID_PK, query_str, primary_key) != 0) {
        std::cerr << "Failed to add into index." << std::endl;
        return -1;
    }
    return 0;
}

const void * IndexBuilder::Dump(size_t *size) {
    const void *data = core_builder_->DumpIndex(size);
    if (data == nullptr) {
        std::cerr << "Failed to Dump index." << std::endl;
        return nullptr;
    }

    return data;
}

int IndexBuilder::DumpRamVamanaIndex(std::string &path_prefix) {
    int ret = core_builder_->DumpRamVamanaIndex(path_prefix);
    if (ret != 0) {
        std::cerr << "Failed to Dump index." << std::endl;
        return -1;
    }
    return ret;
}

int IndexBuilder::MergeShardIndexAndDumpDisk(std::string &path_vamana, size_t* size_vamana, 
                                                std::string &path_medoids, size_t* size_medoids, 
                                                const std::vector<std::string> &shardIndexFiles, 
                                                const std::vector<std::string> &shardIdmapFiles, 
                                                std::string &ori_data_path) {
    int ret = core_builder_->MergeShardIndexAndDumpDisk(path_vamana, size_vamana, 
                                                        path_medoids, size_medoids, 
                                                        shardIndexFiles, shardIdmapFiles, ori_data_path);
    if (ret != 0) {
        std::cerr << "Failed to Dump index." << std::endl;
        return -1;
    }
    return ret;
}

int IndexBuilder::MultiPartSingleDump(std::string &path_vamana, size_t* size_vamana, 
                            std::string &path_pq, size_t* size_pq,
                            std::string &path_medoids, size_t* size_medoids) {
    int ret = core_builder_->MultiPartSingleDump(path_vamana, size_vamana, path_pq, 
                                                size_pq, path_medoids, size_medoids);
    if (ret != 0) {
        std::cerr << "Failed to Dump index." << std::endl;
        return -1;
    }
    return ret;
}

int IndexBuilder::DumpDiskLocal(std::string &path_vamana, size_t* size_vamana) {
    int ret = core_builder_->DumpDiskLocal(path_vamana, size_vamana);
    if (ret != 0) {
        std::cerr << "Failed to Dump index." << std::endl;
        return -1;
    }
    return ret;
}

int IndexBuilder::DumpShardDiskIndex(std::string &shardToken, 
                                     std::string &index_path, size_t* index_size, 
                                     std::string &data_path, size_t* data_size) {
    int ret = core_builder_->DumpShardDiskIndex(shardToken, index_path, index_size, data_path, data_size);
    if (ret != 0) {
        std::cerr << "Failed to Dump index." << std::endl;
        return -1;
    }
    return ret;
}

int IndexBuilder::DumpShardData(std::string &path_prefix) {
    int ret = core_builder_->DumpShardData(path_prefix);
    if (ret <= 0) {
        std::cerr << "Failed to Dump index." << std::endl;
        return -1;
    }
    return ret;
}

// Get vamana shard count
int IndexBuilder::GetShardCount() {
    int ret = core_builder_->GetShardCount();
    if (ret <= 0) {
        std::cerr << "Failed to GetShardCount." << std::endl;
        return -1;
    }
    return ret;
}

int IndexBuilder::AddOriData(const void *val) {
    int ret = core_builder_->AddOriData(val);
    if (ret != 0) {
        std::cerr << "Failed to add ori data." << std::endl;
        return -1;
    }
    return ret;
}

int IndexBuilder::AddRawData(const void *val) {
    int ret = core_builder_->AddRawData(val);
    if (ret != 0) {
        std::cerr << "Failed to add raw data." << std::endl;
        return -1;
    }
    return ret;
}

int IndexBuilder::AddShardData(int shard, RedIndexDocid doc_id, const void *val) {
    int ret = core_builder_->AddShardData(shard, doc_id, val);
    if (ret != 0) {
        std::cerr << "Failed to add shard data." << std::endl;
        return -1;
    }
    return ret;
}

mercury::core::Index::Pointer IndexBuilder::GetIndex() {
    return core_builder_->GetIndex();
}

int64_t IndexBuilder::UsedMemoryInInit() const {
    //TODO why
    return 0x100000;
}

int64_t IndexBuilder::UsedMemoryInCurrent() const {
    return static_cast<int64_t>(core_builder_->GetIndex()->UsedMemoryInCurrent());
}

int64_t IndexBuilder::UsedMemoryInDump() const {
    return static_cast<int64_t>(core_builder_->GetIndex()->UsedMemoryInCurrent());
}

} // namespace redindex
} // namespace mercury
