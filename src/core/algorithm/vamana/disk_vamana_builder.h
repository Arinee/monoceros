# pragma once

#include "src/core/framework/index_framework.h"
#include "src/core/algorithm/builder.h"
#include "disk_vamana_index.h"
#include "bthread/bthread.h"
#include "src/core/algorithm/query_info.h"
#include <omp.h>

MERCURY_NAMESPACE_BEGIN(core);

class DiskVamanaBuilder : public Builder
{

public:
    //! Index Builder Pointer
    typedef std::shared_ptr<DiskVamanaBuilder> Pointer;

    DiskVamanaBuilder();

    //! Initialize Builder
    int Init(IndexParams &params) override;

    //! Build the index
    int AddDoc(docid_t doc_id, uint64_t pk,
               const std::string& build_str, 
               const std::string& primary_key = "") override;

    int BatchProcess();

    int GetRankScore(const std::string& build_str, float * score) override;

    //! Dump index into file
    int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) override;

    //! Dump index into memory
    const void * DumpIndex(size_t* size) override;

    // Build multi-partition vamana index on single machine and dump to disk
    int MultiPartSingleDump(std::string &path_vamana, size_t* size_vamana, 
                            std::string &path_pq, size_t* size_pq,
                            std::string &path_medoids, size_t* size_medoids) override;

    // Dump single partition disk vamana index to local
    int DumpDiskLocal(std::string &path_vamana, size_t* size_vamana) override;

    int MergeShardIndexAndDumpDisk(std::string &path_vamana, size_t* size_vamana, 
                                   std::string &path_medoids, size_t* size_medoids, 
                                   const std::vector<std::string> &shardIndexFiles, 
                                   const std::vector<std::string> &shardIdmapFiles,
                                   std::string &ori_data_path) override;

    // Dump vamana shard index with idmap or pq index with ori-data
    int DumpShardDiskIndex(std::string &shardToken, 
                           std::string &index_path, size_t* index_size, 
                           std::string &data_path, size_t* data_size) override;

    // Dump vamana shard data to local (return shard count)
    int DumpShardData(std::string &path_prefix) override;

    // Get vamana shard count
    int GetShardCount() override;

    int AddOriData(const void *val) override;

    int AddRawData(const void *val) override;

    int AddShardData(int shard, docid_t doc_id, const void *val) override;

    Index::Pointer GetIndex() override {
        return index_;
    }

private:
    DiskVamanaIndex::Pointer index_;

    struct BthreadMessage {
        DiskVamanaIndex::Pointer index;
        std::string ori_filename;
        std::string dump_filename;
    };

    static void* BthreadRun(void* message);

    // (default is false)
    bool parallel_build_;

    // (default is false)
    bool partition_build_;

    struct BatchQueryInfo {
        docid_t docid;
        uint64_t pk;
        QueryInfo * query_info;
        QueryInfo * query_info_raw;
        bool status;
    };

    vector<BatchQueryInfo *> batch_query_;

    // (default is 1000000) num of raw query to process as batch
    size_t batch_count_;

    // (default is 1): num of threads to process raw query
    uint32_t process_thread_num_;
};


MERCURY_NAMESPACE_END(core);
