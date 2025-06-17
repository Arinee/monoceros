/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-14 11:19

#pragma once

#include "redindex_common.h"
#include "src/core/algorithm/builder.h"

namespace mercury {
namespace redindex {

class IndexBuilder {
public:
    using Pointer = std::shared_ptr<IndexBuilder>;
    IndexBuilder(mercury::core::Builder::Pointer core_builder)
        : core_builder_(core_builder){
    }

    int PreAdd(const std::string& build_str, float *score);

    int Add(RedIndexDocid redindex_docid,
            const std::string& query_str, 
            const std::string& primary_key = "");

    const void * Dump(size_t *size);

    int DumpRamVamanaIndex(std::string &path_prefix);

    int DumpDiskLocal(std::string &path_vamana, size_t* size_vamana);

    // Build multi-partition vamana index on single machine and dump to disk
    int MultiPartSingleDump(std::string &path_vamana, size_t* size_vamana, 
                            std::string &path_pq, size_t* size_pq,
                            std::string &path_medoids, size_t* size_medoids);

    int MergeShardIndexAndDumpDisk(std::string &path_vamana, size_t* size_vamana, 
                                    std::string &path_medoids, size_t* size_medoids, 
                                    const std::vector<std::string> &shardIndexFiles, 
                                    const std::vector<std::string> &shardIdmapFiles,
                                    std::string &ori_data_path);

    // Dump vamana shard index with idmap or pq index with ori-data
    int DumpShardDiskIndex(std::string &shardToken, 
                           std::string &index_path, size_t* index_size, 
                           std::string &data_path, size_t* data_size);

    int DumpShardData(std::string &path_prefix);

    // Get vamana shard count
    int GetShardCount();

    int AddOriData(const void *val);

    int AddRawData(const void *val);

    int AddShardData(int shard, RedIndexDocid redindex_docid, const void *val);

    mercury::core::Index::Pointer GetIndex();

    int64_t UsedMemoryInInit() const;
    int64_t UsedMemoryInCurrent() const;
    int64_t UsedMemoryInDump() const;

private:
    mercury::core::Builder::Pointer core_builder_;
};

} // namespace redindex
} // namespace mercury
