#ifndef __MERCURY_CORE_INDEX_BUILDER_H__
#define __MERCURY_CORE_INDEX_BUILDER_H__

#include <memory>
#include "src/core/framework/index_storage.h"
#include "src/core/common/common.h"
#include "index.h"

MERCURY_NAMESPACE_BEGIN(core);

/*! Index Builder
 */
class Builder
{
public:
    //! Index Builder Pointer
    typedef std::shared_ptr<Builder> Pointer;

    //! Destructor
    //virtual ~Builder() = default;

    //! Initialize Builder
    virtual int Init(IndexParams &params) = 0;

    //! Build the index
    virtual int AddDoc(docid_t doc_id, uint64_t pk,
                       const std::string& build_str, 
                       const std::string& primary_key = "") = 0;

    virtual int GetRankScore(const std::string& query_str, float * score) = 0;

    //! Dump index into file or memory
    virtual int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) = 0;

    virtual const void * DumpIndex(size_t* size) = 0;

    // Dump ram vamana index
    virtual int DumpRamVamanaIndex(std::string &path_prefix) {
        return -1;
    }

    // Build multi-partition vamana index on single machine and dump to disk
    virtual int MultiPartSingleDump(std::string &path_vamana, size_t* size_vamana, 
                                    std::string &path_pq, size_t* size_pq,
                                    std::string &path_medoids, size_t* size_medoids) {
        return -1;
    }

    // Dump single partition disk vamana index to local
    virtual int DumpDiskLocal(std::string &path_vamana, size_t* size_vamana) {
        return -1;
    }

    virtual int MergeShardIndexAndDumpDisk(std::string &path_vamana, size_t* size_vamana, 
                                            std::string &path_medoids, size_t* size_medoids, 
                                            const std::vector<std::string> &shardIndexFiles, 
                                            const std::vector<std::string> &shardIdmapFiles,
                                            std::string &ori_data_path) {
        return -1;
    }

    // Dump vamana shard index with idmap or pq index with ori-data
    virtual int DumpShardDiskIndex(std::string &shardToken, 
                                   std::string &index_path, size_t* index_size, 
                                   std::string &data_path, size_t* data_size) 
    {
        return -1;
    }

    // Dump vamana shard data to local (return shard count)
    virtual int DumpShardData(std::string &path_prefix) {
        path_prefix = "";
        return -1;
    }

    // Get vamana shard count
    virtual int GetShardCount() {
        return -1;
    }

    virtual int AddOriData(const void *val) {
        return -1;
    }

    virtual int AddRawData(const void *val) {
        return -1;
    }

    virtual int AddShardData(int shard, docid_t doc_id, const void *val) {
        return -1;
    }

    virtual Index::Pointer GetIndex() = 0;
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_CORE_INDEX_BUILDER_H__
