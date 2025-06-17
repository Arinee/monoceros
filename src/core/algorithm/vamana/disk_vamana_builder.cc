#include "disk_vamana_builder.h"
#include "src/core/common/common_define.h"

MERCURY_NAMESPACE_BEGIN(core);

DiskVamanaBuilder::DiskVamanaBuilder()
    : parallel_build_(false),
      partition_build_(false),
      batch_count_(1000000),
      process_thread_num_(1)
{}

int DiskVamanaBuilder::Init(IndexParams &params) {
    if (!index_) {
        index_.reset(new DiskVamanaIndex());
    }

    if (index_->Create(params) != 0) {
        LOG_ERROR("Failed to init disk vamana index.");
        return -1;
    }

    if (params.getString(PARAM_DISKANN_BUILD_MODE) == "parallel") {
        parallel_build_ = true;
    }
    
    if (params.getBool(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION)) {
        partition_build_ = true;
    }

    uint32_t build_thread_num = params.getUint32(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM);
    if (build_thread_num <= 0) {
        LOG_ERROR("mercury.vamana.index.build.thread_num must larger than 0");
        return -1;
    }

    process_thread_num_ = (build_thread_num / 2) != 0 ? (build_thread_num / 2) : 1;

    batch_count_ = params.getUint64(PARAM_VAMANA_INDEX_BUILD_BATCH_COUNT);
    if (batch_count_ <= 0) {
        LOG_ERROR("mercury.vamana.index.build.batch_count must larger than 0");
        return -1;
    }

    return 0;
}

//! Build the index
int DiskVamanaBuilder::AddDoc(docid_t doc_id, uint64_t pk,
                             const std::string& build_str, 
                             const std::string& primary_key) {

    QueryInfo * query = new QueryInfo(build_str);
    BatchQueryInfo * batchQueryInfo = new BatchQueryInfo();
    batchQueryInfo->docid = doc_id;
    batchQueryInfo->pk = pk;
    batchQueryInfo->query_info = query;
    batchQueryInfo->query_info_raw = nullptr;
    batchQueryInfo->status = false;
    batch_query_.push_back(batchQueryInfo);

    int ret = 0;

    if (batch_query_.size() == batch_count_) {
        ret = BatchProcess();
    }

    return ret;
}

int DiskVamanaBuilder::BatchProcess() {

    LOG_INFO("Call batch process with %lu records and %d threads", batch_query_.size(), process_thread_num_);

    auto start = std::chrono::high_resolution_clock::now();

    omp_set_num_threads(process_thread_num_);

#pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < batch_query_.size(); i++) {
        QueryInfo * query = batch_query_[i]->query_info;

        batch_query_[i]->status = true;

        if (index_->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
            query->SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
            QueryInfo * query_raw = new QueryInfo(query->GetRawQuery());
            if (!query_raw->MakeAsBuilder()) {
                LOG_ERROR("resolve query_raw failed. query_raw str:%s", query_raw->GetRawQuery().c_str());
                batch_query_[i]->status = false;
            }
            batch_query_[i]->query_info_raw = query_raw;
        }

        if (!query->MakeAsBuilder()) {
            LOG_ERROR("resolve query failed. query str:%s", query->GetRawQuery().c_str());
            batch_query_[i]->status = false;
        }
    }

    for (auto batch_query_info : batch_query_) {
        if (!batch_query_info->status) {
            LOG_ERROR("Failed to process docid: %d", batch_query_info->docid);
            return -1;
        }
        docid_t cur_docid = batch_query_info->docid;
        uint64_t cur_pk = batch_query_info->pk;
        QueryInfo * query = batch_query_info->query_info;
        QueryInfo * query_raw = batch_query_info->query_info_raw;
        if (partition_build_) {
            if (index_->PartitionBaseIndexAdd(cur_docid, cur_pk, *query, *query_raw) != 0) {
                return -1;
            }
        } else {
            if (index_->BaseIndexAdd(cur_docid, cur_pk, query->GetVector(), query->GetVectorLen()) != 0) {
                return -1;
            }
            if (index_->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
                if (index_->PQIndexAdd(cur_docid, query_raw->GetVector()) != 0) {
                    return -1;
                }
            } else {
                if (index_->PQIndexAdd(cur_docid, query->GetVector()) != 0) {
                    return -1;
                }
            }
        }
        if (query != nullptr) {
            delete query;
            query = nullptr;
        }
        if (query_raw != nullptr) {
            delete query_raw;
            query_raw = nullptr;
        }
        if (batch_query_info != nullptr) {
            delete batch_query_info;
            batch_query_info = nullptr;
        }
    }
    batch_query_.clear();
    batch_query_.shrink_to_fit();

    auto elapsed = std::chrono::high_resolution_clock::now() - start;

    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

    LOG_INFO("Process finished with %fs", (float)microseconds / 1000000);

    return 0;
}

//! Dump index into file or memory
int DiskVamanaBuilder::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) {
    //no use
    return 0;
}

const void * DiskVamanaBuilder::DumpIndex(size_t* size) {

    if (batch_query_.size() != 0) {
        int ret = BatchProcess();
        if (ret != 0) {
            LOG_ERROR("Failed to process last batch of query");
            return nullptr;
        }
    }

    if (!index_->IsPqTablePopulated()) {
        LOG_ERROR("Pq table not been populated for disk vamana index.");
        return nullptr;
    }

    // dump index to a data pointer and return 
    const void *data = nullptr;
    if (index_->Dump(data, *size) != 0) {
        LOG_ERROR("Failed to dump index.");
        *size = 0;
        return nullptr;
    }

    return data;
}

void* DiskVamanaBuilder::BthreadRun(void* message) {
    BthreadMessage* msg = static_cast<BthreadMessage*>(message);
    msg->index->BuildPqIndexFromFile(msg->ori_filename);
    if (!msg->index->IsPqTablePopulated()) {
        LOG_ERROR("Pq table not been populated for disk vamana index");
        return nullptr;
    }
    int ret = msg->index->DumpPqLocal(msg->dump_filename);
    if (ret != 0) {
        LOG_ERROR("Pq table dump failed");
        return nullptr;
    }
    return nullptr;
}

int DiskVamanaBuilder::MultiPartSingleDump(std::string &path_vamana, size_t* size_vamana, 
                                           std::string &path_pq, size_t* size_pq,
                                           std::string &path_medoids, size_t* size_medoids) {
    if (batch_query_.size() != 0) {
        int ret = BatchProcess();
        if (ret != 0) {
            LOG_ERROR("Failed to process last batch of query");
            return -1;
        }
    }
    
    index_->PartitionBaseIndexDump();
    std::string vamana_partition_ori_data_path = index_->_partition_prefix + "_ori.data";
    std::string vamana_partition_raw_data_path = index_->_partition_prefix + "_raw.data";
    std::string vamana_partition_merged_mem_index_path = index_->_partition_prefix + "_merged_mem.index";
    std::string vamana_partition_disk_index_path = index_->_partition_prefix + "_disk.index";
    std::string vamana_partition_pq_index_path = index_->_partition_prefix + "_pq.index";
    bthread_t bid;
    BthreadMessage msg;
    msg.index = index_;
    if (index_->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
        msg.ori_filename = vamana_partition_raw_data_path;
    } else {
        msg.ori_filename = vamana_partition_ori_data_path;
    }
    msg.dump_filename = vamana_partition_pq_index_path;
    if (bthread_start_urgent(&bid, NULL, BthreadRun, &msg) != 0) {
        LOG_ERROR("start bthread failed.");
        return -1;
    }
    index_->BuildAndDumpPartitionIndex();
    bthread_join(bid, NULL);
    index_->MergePartitionIndex(path_medoids, size_medoids);
    index_->CreateDiskLayout(vamana_partition_ori_data_path, vamana_partition_merged_mem_index_path, vamana_partition_disk_index_path);
    if (file_exists(vamana_partition_disk_index_path)) {
        path_vamana = vamana_partition_disk_index_path;
        *size_vamana = get_file_size(vamana_partition_disk_index_path);
    } else {
        LOG_ERROR("Dump vamana disk index failed");
        return -1;
    }
    if (file_exists(vamana_partition_pq_index_path)) {
        path_pq = vamana_partition_pq_index_path;
        *size_pq = get_file_size(vamana_partition_pq_index_path);
    } else {
        LOG_ERROR("Dump vamana pq index failed");
        return -1;
    }
    return 0;
}

int DiskVamanaBuilder::DumpDiskLocal(std::string &path_vamana, size_t* size_vamana) {
    if (batch_query_.size() != 0) {
        int ret = BatchProcess();
        if (ret != 0) {
            LOG_ERROR("Failed to process last batch of query");
            return -1;
        }
    }
    std::string vamana_mem_index_path = index_->_partition_prefix + "_mem.index";
    std::string vamana_disk_index_path = index_->_partition_prefix + "_disk.index";
    index_->BuildMemIndex();
    index_->DumpMemLocal(vamana_mem_index_path);
    index_->CreateDiskLayout(vamana_mem_index_path + ".data", vamana_mem_index_path, vamana_disk_index_path);
    if (file_exists(vamana_disk_index_path)) {
        path_vamana = vamana_disk_index_path;
        *size_vamana = get_file_size(vamana_disk_index_path);
    } else {
        LOG_ERROR("Dump vamana disk index failed");
        return -1;
    }
    return 0;
}

int DiskVamanaBuilder::MergeShardIndexAndDumpDisk(std::string &path_vamana, size_t* size_vamana, 
                                                  std::string &path_medoids, size_t* size_medoids, 
                                                  const std::vector<std::string> &shardIndexFiles, 
                                                  const std::vector<std::string> &shardIdmapFiles,
                                                  std::string &ori_data_path) 
{
    index_->MergeShardIndex(path_medoids, size_medoids, shardIndexFiles, shardIdmapFiles);
    std::string vamana_partition_merged_mem_index_path = index_->_partition_prefix + "_merged_mem.index";
    std::string vamana_partition_disk_index_path = index_->_partition_prefix + "_disk.index";
    index_->CreateDiskLayout(ori_data_path, vamana_partition_merged_mem_index_path, vamana_partition_disk_index_path);
    if (file_exists(vamana_partition_disk_index_path)) {
        path_vamana = vamana_partition_disk_index_path;
        *size_vamana = get_file_size(vamana_partition_disk_index_path);
    } else {
        LOG_ERROR("Dump vamana disk index failed");
        return -1;
    }
    return 0;
}

int DiskVamanaBuilder::DumpShardDiskIndex(std::string &shardToken, 
                                          std::string &index_path, size_t* index_size, 
                                          std::string &data_path, size_t* data_size) {
    if (batch_query_.size() != 0) {
        int ret = BatchProcess();
        if (ret != 0) {
            LOG_ERROR("Failed to process last batch of query");
            return -1;
        }
    }
    int ret = index_->ShardDataDump(shardToken, data_path);
    if (ret != 0) {
        LOG_ERROR("Failed to dump shard data");
        return -1;
    }
    *data_size = get_file_size(data_path);
    ret = index_->ShardIndexBuildAndDump(shardToken, index_path);
    if (ret != 0) {
        LOG_ERROR("Failed to build and dump shard data index");
        return -1;
    }
    *index_size = get_file_size(index_path);
    return 0;
}

int DiskVamanaBuilder::DumpShardData(std::string &path_prefix) {
    if (batch_query_.size() != 0) {
        int ret = BatchProcess();
        if (ret != 0) {
            LOG_ERROR("Failed to process last batch of query");
            return -1;
        }
    }
    
    index_->PartitionBaseIndexDump();

    path_prefix = index_->_partition_prefix;

    return index_->GetCenterNum();
}

int DiskVamanaBuilder::GetShardCount() {
    return index_->GetCenterNum();
}

int DiskVamanaBuilder::AddOriData(const void *val) {
    return index_->AddOriData(val);
}

int DiskVamanaBuilder::AddRawData(const void *val) {
    return index_->AddRawData(val);
}

int DiskVamanaBuilder::AddShardData(int shard, docid_t doc_id, const void *val) {
    return index_->AddShardData(shard, doc_id, val);
}

int DiskVamanaBuilder::GetRankScore(const std::string& build_str, float * score) {
    return 0;
}

MERCURY_NAMESPACE_END(core);
