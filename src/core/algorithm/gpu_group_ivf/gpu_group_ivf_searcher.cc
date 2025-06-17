#ifdef ENABLE_GPU_IN_MERCURY_
#include "gpu_group_ivf_searcher.h"
#include "bthread/bthread.h"
#include "putil/mem_pool/PoolVector.h"
#include "src/core/algorithm/thread_common.h"
#include "src/core/framework/utility/closure.h"

DEFINE_int32(max_batch_size, 5, "flag for max_batch_size");
DEFINE_uint64(batch_timeout_micros, 1000, "flag for batch_timeout_micros");
DEFINE_uint32(doc_num_per_tile, 10240, "flag for doc_num_per_tile");
DEFINE_uint32(doc_num_per_block, 512, "flag for doc_num_per_block");
DEFINE_int32(num_batch_threads, 1, "flag for num_batch_threads");
DEFINE_uint64(periodic_interval_micros, 0, "flag for periodic_interval_micros");
DEFINE_uint32(batch_scheduler_num, 1, "flag for batch_scheduler_num");
DEFINE_uint32(batch_scheduler_doc_num_threshold, 50000, "flag for batch_scheduler_doc_num_threshold");
DEFINE_uint32(batch_scheduler_doc_num_split, 50000, "flag for batch_scheduler_doc_num_split");
DEFINE_uint32(max_batch_concurrency_num, 30, "flag for batch_scheduler_doc_num_threshold");
DEFINE_uint32(max_enqueued_batch_num, 1000, "flag for max_enqueued_batch_num");
// DEFINE_uint32(batch_scheduler_group_num_threshold, 30,
//               "flag for batch_scheduler_group_num_threshold");
DEFINE_uint32(global_batch_force_control, 0,
              "flag for global_batch_force_control"); // 0: no control, 1: force
                                                      // enable, 2: force disable
DEFINE_uint32(global_half_force_control, 0,
              "flag for global_batch_force_control"); // 0: no control, 1: force
                                                      // enable, 2: force disable
DEFINE_uint32(global_enable_device_count, 1, "flag for global_enable_device_count");

MERCURY_NAMESPACE_BEGIN(core);

size_t
GpuGroupIvfSearcher::GetPostingsNodeCount(const std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings,
                                          size_t start, size_t end) const
{
    size_t count = 0;
    for (size_t i = start; i < end; i++) {
        count += ivf_postings.at(i).getDocNum();
    }

    return count;
}

GpuGroupIvfSearcher::GpuGroupIvfSearcher() : random_(19820606)
{
    if (!index_) {
        index_.reset(new GroupIvfIndex());
    }
    SetThreadEnv();
}

GpuGroupIvfSearcher::~GpuGroupIvfSearcher()
{
    if (is_gpu_rt_ && gpu_index_) {
        delete gpu_index_;
        gpu_index_ = nullptr;
        LOG_INFO("delete rt gpu_index_ finished, %s", index_name_.c_str());
    } else if (own_gpu_index_ && gpu_index_) {
        if (GpuResourcesWrapper::GetInstance()->GetIndexGpuRecord(gpu_record_key_)) {
            if (GpuResourcesWrapper::GetInstance()->RemoveIndexGpuRecord(gpu_record_key_)) {
                // 真正被删掉了，才做析构
                delete gpu_index_;
                gpu_index_ = nullptr;
                LOG_INFO("delete gpu_index_ finished, %s", gpu_record_key_.c_str());
            }
        }
    }
    LOG_INFO("GpuGroupIvfSearcher destructor finished, %s", index_name_.c_str());
}

int GpuGroupIvfSearcher::Init(IndexParams &params)
{
    index_->SetIndexParams(params);
    index_->SetIsGpu(true);

    std::string index_name = params.getString(PARAM_VECTOR_INDEX_NAME);
    LOG_INFO("Start Init GpuGroupIvfSearcher, %s", index_name.c_str());

    MONITOR_METRIC(GpuGroupIvf_CentroidNum);
    MONITOR_METRIC_WITH_INDEX(GpuGroupIvf_GroupNum, "GpuGroupIvf_GroupNum_" + index_name);
    MONITOR_METRIC_WITH_INDEX(GpuGroupIvf_FullDocNum, "GpuGroupIvf_FullDocNum_" + index_name);
    MONITOR_METRIC_WITH_INDEX(GpuGroupIvf_RtDocNum, "GpuGroupIvf_RtDocNum_" + index_name);
    MONITOR_METRIC(GpuGroupIvf_FullResultDocNum);
    MONITOR_METRIC(GpuGroupIvf_RtResultDocNum);
    MONITOR_METRIC(GpuGroupIvf_BatchConcurrencyNum)
    MONITOR_TRANSACTION(GpuGroupIvf, GpuBatchSize);
    MONITOR_TRANSACTION(GpuGroupIvf, GpuProcessTime);
    MONITOR_TRANSACTION(GpuGroupIvf, GpuPreProcessTime);
    MONITOR_TRANSACTION(GpuGroupIvf, GpuPostProcessTime);
    MONITOR_TRANSACTION(GpuGroupIvf, GpuBatchPreProcessTime);
    MONITOR_TRANSACTION(GpuGroupIvf, GpuBatchProcessTime);
    MONITOR_TRANSACTION(GpuGroupIvf, GpuBatchProcessNotifyTime);
    MONITOR_TRANSACTION(GpuGroupIvf, GpuCollectIvfNextTime);
    MONITOR_TRANSACTION(GpuGroupIvf, GpuBatchFirstTaskWaitTime);
    MONITOR_TRANSACTION(GpuGroupIvf, GpuBatchLastTaskWaitTime);
    MONITOR_TRANSACTION(GpuGroupIvf, GpuBatchAllTaskWaitTime);
    MONITOR_TRANSACTION(GpuGroupIvf, GpuBatchDataSize);
    index_name_ = index_name;

    char *pEnd;
    sort_mode_ = std::strtol(std::getenv("PARAM_SORT_MODE"), &pEnd, 10);

    LOG_INFO("End Init GpuGroupIvfSearcher, %s", index_name.c_str());
    return 0;
}

int GpuGroupIvfSearcher::LoadIndex(const std::string &path)
{
    // TODO
    return -1;
}

int GpuGroupIvfSearcher::LoadIndex(const void *data, size_t size)
{
    gpu_record_key_ = index_name_ + std::to_string(reinterpret_cast<uint64_t>(data));
    if (index_->Load(data, size) != 0) {
        LOG_ERROR("Failed to load index.");
        return -1;
    }
    // 获取向量地址
    const void *base = nullptr;
    if (!index_->ContainFeature()) {
        if (!vector_retriever_.isValid()) {
            LOG_ERROR("[%s] can not find vector in array profile or attribute",
                      index_->GetIndexParams().getString(PARAM_VECTOR_INDEX_NAME).c_str());
            return -1;
        } else if (!vector_retriever_.isLengthFixed()) {
            LOG_ERROR("[%s] find vector in attribute but not fixed length",
                      index_->GetIndexParams().getString(PARAM_VECTOR_INDEX_NAME).c_str());
            return -1;
        } else {
            vector_retriever_(0, base);
        }
    } else {
        base = (const void *)index_->GetFeatureProfile().getInfo(0);
    }
    const IndexMeta &index_meta = index_->GetIndexMeta();
    dist_scorer_factory_.Init(index_meta);

    // 索引名
    std::string index_name = index_->GetIndexParams().getString(PARAM_VECTOR_INDEX_NAME);

    // 放在哪张卡上
    uint32_t device_no = index_->GetIndexParams().has(PARAM_VECTOR_ENABLE_DEVICE_NO)
                             ? index_->GetIndexParams().getUint32(PARAM_VECTOR_ENABLE_DEVICE_NO)
                             : 0;
    if (device_no >= FLAGS_global_enable_device_count) {
        LOG_INFO("index name: %s, device_no %u is larger than global_enable_device_count %u, rewrite to 0",
                 index_name.c_str(), device_no, FLAGS_global_enable_device_count);
        device_no = 0;
    }

    // 初始化gpu searcher (dim 64, feature_size 64*4)
    if (GpuResourcesWrapper::GetInstance()->GetIndexGpuRecord(gpu_record_key_)) {
        gpu_index_ = (neutron::gpu::NeutronIndexInterface *)(GpuResourcesWrapper::GetInstance()->GetIndexGpuRecord(
            gpu_record_key_));
        GpuResourcesWrapper::GetInstance()->IncIndexGpuRecord(gpu_record_key_);
        LOG_INFO("Neutron gpu_index duplicated create, use the previous one, %s", gpu_record_key_.c_str());
    } else {
        if (index_meta.type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
            // 索引是半精度的
            LOG_INFO("Created Neutron Index with direct half: %s", index_name.c_str());
            gpu_index_ =
                new neutron::gpu::NeutronIndexInterface(neutron::gpu::NeutronIndexInterface::CALCULATORTYPE::DIRECTHALF,
                                                        index_meta.dimension(), index_meta.sizeofElement(), device_no);
        } else {
            // 索引是float的, 需要转换成半精度的
            if (FLAGS_global_half_force_control != 2 &&
                ((FLAGS_global_half_force_control == 1) ||
                 (index_->GetIndexParams().has(PARAM_VECTOR_ENABLE_HALF) &&
                  index_->GetIndexParams().getBool(PARAM_VECTOR_ENABLE_HALF)))) {
                LOG_INFO("Created Neutron Index with half float: %s", index_name.c_str());
                gpu_index_ = new neutron::gpu::NeutronIndexInterface(
                    neutron::gpu::NeutronIndexInterface::CALCULATORTYPE::HALF, index_meta.dimension(),
                    index_meta.sizeofElement() / 2, device_no);
            } else {
                LOG_INFO("Created Neutron Index with float: %s", index_name.c_str());
                gpu_index_ = new neutron::gpu::NeutronIndexInterface(
                    neutron::gpu::NeutronIndexInterface::CALCULATORTYPE::FLOAT, index_meta.dimension(),
                    index_meta.sizeofElement(), device_no);
            }
        }

        // 通用
        neutron_manager_interface_ = GpuResourcesWrapper::GetInstance()->GetNeutronManagerInterface();
        gpu_index_->SetNeutronManagerInterface(neutron_manager_interface_);
        // 将索引中的向量全部拷贝到gpu device中 (feature, num 500)
        gpu_index_->Add((const char *)base, index_->GetDocNum());
        GpuResourcesWrapper::GetInstance()->AddIndexGpuRecord(gpu_record_key_, gpu_index_);
    }

    own_gpu_index_ = true;

    // 配置是否使用batch
    if (FLAGS_global_batch_force_control != 2 &&
        ((FLAGS_global_batch_force_control == 1) || (index_->GetIndexParams().has(PARAM_VECTOR_ENABLE_BATCH) &&
                                                     index_->GetIndexParams().getBool(PARAM_VECTOR_ENABLE_BATCH)))) {
        enable_batch_ = true;
        BasicBatchScheduler<GpuGroupIvfBatchTask>::Options option;
        option.max_batch_size = FLAGS_max_batch_size;
        option.batch_timeout_micros = FLAGS_batch_timeout_micros;
        option.num_batch_threads = FLAGS_num_batch_threads;
        option.periodic_interval_micros = FLAGS_periodic_interval_micros;
        option.max_batch_concurrency_num = FLAGS_max_batch_concurrency_num;
        option.max_enqueued_batches = FLAGS_max_enqueued_batch_num;
        auto callback = [&](std::shared_ptr<Batch<GpuGroupIvfBatchTask>> batch) {
            butil::Timer timer;
            timer.start();
            transaction_GpuBatchSize(batch->size() * 1000, true);
            uint64_t first_task_wait_time = batch->task_start_process_time_ - batch->task_add_time_.front();
            transaction_GpuBatchFirstTaskWaitTime(first_task_wait_time, true);
            uint64_t last_task_wait_time = batch->task_start_process_time_ - batch->task_add_time_.back();
            transaction_GpuBatchLastTaskWaitTime(last_task_wait_time, true);
            uint64_t avg_time = 0;
            for (auto &add_time : batch->task_add_time_) {
                avg_time += batch->task_start_process_time_ - add_time;
            }
            avg_time /= batch->task_add_time_.size();
            transaction_GpuBatchAllTaskWaitTime(avg_time, true);

            GpuGroupIvfBatchTask *first_task = batch->mutable_task(0);
            putil::mem_pool::Pool *pool = first_task->pool_;

            // 目前就存了4种字段，只需要4个offset即可
            putil::mem_pool::PoolVector<uint64_t> data_offsets(pool, 8, 0);
            // 1. 计算所需字节数
            uint64_t allocated_bytes = 0;
            uint32_t task_num = batch->num_tasks();
            // query, only support float now
            uint64_t query_size = index_->GetIndexMeta().sizeofElement();
            allocated_bytes += task_num * query_size;
            // distance compute, calculate (512 aligned) doc_num &&
            uint64_t aligned_doc_num = 0;
            uint32_t max_topk = 0;
            putil::mem_pool::PoolVector<uint32_t> start_ends(pool, task_num * 4, 0);
            uint32_t sort_block_num = 0;
            uint32_t batch_group_num = 0;
            bool enable_cate = false;
            bool enable_rt = false;
            uint32_t max_len = 0;

            for (uint32_t i = 0; i < task_num; i++) {
                GpuGroupIvfBatchTask *task = batch->mutable_task(i);
                max_len = std::max<uint32_t>(max_len, task->dist_nodes_->size());
                start_ends[i * 4] = aligned_doc_num;
                start_ends[i * 4 + 1] = aligned_doc_num + task->dist_nodes_->size();
                start_ends[i * 4 + 2] = aligned_doc_num + (task->dist_nodes_->size() + 511) / 512 * 512;
                start_ends[i * 4 + 3] = sort_block_num + (task->dist_nodes_->size() + 10239) / 10240;
                aligned_doc_num = start_ends[i * 4 + 2];
                sort_block_num = start_ends[i * 4 + 3];
                max_topk = std::max(max_topk, task->total_topk_);
                batch_group_num += task->group_num_;
                enable_cate = enable_cate || task->enable_cate_;
                enable_rt = enable_rt || task->enable_rt_;
            }
            // start end
            data_offsets[0] = allocated_bytes;
            allocated_bytes += task_num * 4 * sizeof(uint32_t);
            // docid
            data_offsets[1] = allocated_bytes;
            allocated_bytes += aligned_doc_num * sizeof(uint32_t);
            // result
            data_offsets[2] = allocated_bytes;
            if (enable_cate) {
                allocated_bytes += batch_group_num * max_topk * (sizeof(float) + sizeof(uint32_t));
            } else {
                allocated_bytes += task_num * max_topk * (sizeof(float) + sizeof(uint32_t));
            }
            if (enable_cate) {
                // query_doc_starts
                data_offsets[3] = allocated_bytes;
                allocated_bytes += batch_group_num * sizeof(uint32_t);
                // group_doc_starts
                data_offsets[4] = allocated_bytes;
                allocated_bytes += batch_group_num * sizeof(uint32_t);
                // group_doc_nums
                data_offsets[5] = allocated_bytes;
                allocated_bytes += batch_group_num * sizeof(uint32_t);
                // group_doc_rt_starts
                data_offsets[6] = allocated_bytes;
                allocated_bytes += batch_group_num * sizeof(uint32_t);
                // group_doc_rt_nums
                data_offsets[7] = allocated_bytes;
                allocated_bytes += batch_group_num * sizeof(uint32_t);
            }
            transaction_GpuBatchDataSize(allocated_bytes, true);

            // 2. 预分配 32-byte aligned memory
            char *batch_data = (char *)pool->allocateAlign(allocated_bytes, 32);
            // 3. 写入真实数据
            // query
            char *write_data = batch_data;
            for (uint32_t i = 0; i < task_num; i++) {
                GpuGroupIvfBatchTask *task = batch->mutable_task(i);
                memcpy(write_data, task->query_info_->GetVector(), query_size);
                write_data += query_size;
            }
            // start end
            write_data = batch_data + data_offsets[0];
            memcpy(write_data, start_ends.data(), start_ends.size() * sizeof(uint32_t));
            // docid
            write_data = batch_data + data_offsets[1];
            for (uint32_t i = 0; i < task_num; i++) {
                GpuGroupIvfBatchTask *task = batch->mutable_task(i);
                memcpy((uint32_t *)write_data + start_ends[i * 4], task->dist_nodes_->data(),
                       task->dist_nodes_->size() * sizeof(uint32_t));
                memset((uint32_t *)write_data + start_ends[i * 4 + 1], 0,
                       (start_ends[i * 4 + 2] - start_ends[i * 4 + 1]) * sizeof(uint32_t));
            }
            if (enable_cate) {
                uint32_t *query_doc_starts = (uint32_t *)(batch_data + data_offsets[3]);
                uint32_t *group_doc_starts = (uint32_t *)(batch_data + data_offsets[4]);
                uint32_t *group_doc_nums = (uint32_t *)(batch_data + data_offsets[5]);
                uint32_t *group_doc_rt_starts = (uint32_t *)(batch_data + data_offsets[6]);
                uint32_t *group_doc_rt_nums = (uint32_t *)(batch_data + data_offsets[7]);
                for (uint32_t i = 0; i < task_num; i++) {
                    GpuGroupIvfBatchTask *task = batch->mutable_task(i);
                    uint32_t group_doc_start = 0;
                    for (uint32_t j = 0; j < task->group_num_; j++) {
                        *(query_doc_starts++) = start_ends[i * 4];
                        *(group_doc_starts++) = group_doc_start;
                        group_doc_start += task->group_doc_nums_->at(j);
                        *(group_doc_nums++) = task->group_doc_nums_->at(j);
                    }
                    if (enable_rt) {
                        for (uint32_t j = task->group_num_; j < task->group_num_ * 2; j++) {
                            *(group_doc_rt_starts++) = group_doc_start;
                            group_doc_start += task->group_doc_nums_->at(j);
                            *(group_doc_rt_nums++) = task->group_doc_nums_->at(j);
                        }
                    }
                }
            }

            timer.stop();
            transaction_GpuBatchPreProcessTime(timer.u_elapsed(), true);
            timer.start();
            //  调用gpu batch接口
            int success = -1; // 0 成功, 非0 失败
            if (enable_cate) {
                success = this->gpu_index_->SearchBatchCate(task_num, batch_group_num, max_topk, batch_data,
                                                            data_offsets.data(), allocated_bytes, enable_rt);
            } else {
                success = this->gpu_index_->SearchBatch(task_num, sort_block_num, max_topk, max_len, batch_data,
                                                        data_offsets.data(), allocated_bytes);
            }

            if (success == 0) {
                // 最终结果写回每个task
                // result
                write_data = batch_data + data_offsets[2];
                if (enable_cate) {
                    // TODO: 处理不同topk的问题
                    for (size_t i = 0; i < task_num; i++) {
                        GpuGroupIvfBatchTask *task = batch->mutable_task(i);
                        memcpy(*(task->return_data_), write_data, task->group_num_ * max_topk * 8);
                        write_data += task->group_num_ * max_topk * 8;
                    }
                } else {
                    if (sort_mode_ == 0) {
                        for (size_t i = 0; i < task_num; i++) {
                            GpuGroupIvfBatchTask *task = batch->mutable_task(i);
                            if (task->total_topk_ == max_topk) {
                                memcpy(*(task->return_data_), write_data, max_topk * 8);
                            } else {
                                memcpy(*(task->return_data_), write_data,
                                       task->total_topk_ * sizeof(float)); // distance
                                memcpy(*(task->return_data_) + task->total_topk_ * sizeof(float),
                                       write_data + max_topk * sizeof(float),
                                       task->total_topk_ * sizeof(uint32_t)); // label
                            }
                            write_data += max_topk * 8;
                        }
                    } else {
                        // TODO: 处理不同topk的问题
                        float *out = (float *)write_data;
                        uint32_t *out_idx = (uint32_t *)write_data + max_topk * task_num;
                        for (size_t i = 0; i < task_num; i++) {
                            GpuGroupIvfBatchTask *task = batch->mutable_task(i);
                            memcpy(*(task->return_data_), out,
                                   task->total_topk_ * sizeof(float)); // distance
                            memcpy(*(task->return_data_) + task->total_topk_ * sizeof(float), out_idx,
                                   task->total_topk_ * sizeof(uint32_t)); // label
                            out += max_topk;
                            out_idx += max_topk;
                        }
                    }
                }
            }

            timer.stop();
            transaction_GpuBatchProcessTime(timer.u_elapsed(), success == 0 ? true : false);
            timer.start();

            for (uint32_t i = 0; i < task_num; i++) {
                GpuGroupIvfBatchTask *task = batch->mutable_task(i);
                bthread_mutex_lock(task->batch_mutex_);
                task->is_finish = true;
                task->is_success = (success == 0);
                bthread_cond_signal(task->batch_cond_);
                bthread_mutex_unlock(task->batch_mutex_);
            }
            timer.stop();
            // MONITOR_METRIC_LOG(GpuGroupIvf_BatchConcurrencyNum,
            //                    basic_batch_scheduler_->FinishOneBatch());
            transaction_GpuBatchProcessNotifyTime(timer.u_elapsed(), true);

            return;
        };

        basic_batch_schedulers_.resize(FLAGS_batch_scheduler_num);
        for (size_t i = 0; i < FLAGS_batch_scheduler_num; i++) {
            basic_batch_schedulers_[i].reset(new BasicBatchScheduler<GpuGroupIvfBatchTask>(option, callback));
        }
    }
    return 0;
}

void GpuGroupIvfSearcher::SetIndex(Index::Pointer index)
{
    index_ = std::dynamic_pointer_cast<GroupIvfIndex>(index);
    auto index_meta = index_->GetIndexMeta();
    dist_scorer_factory_.Init(index_meta);

    // 放在哪张卡上
    uint32_t device_no = index_->GetIndexParams().has(PARAM_VECTOR_ENABLE_DEVICE_NO)
                             ? index_->GetIndexParams().getUint32(PARAM_VECTOR_ENABLE_DEVICE_NO)
                             : 0;
    if (device_no >= FLAGS_global_enable_device_count) {
        LOG_INFO("index name: %s, device_no %u is larger than global_enable_device_count %u, rewrite to 0",
                 index_name_.c_str(), device_no, FLAGS_global_enable_device_count);
        device_no = 0;
    }

    if (index_meta.type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
        // 索引是半精度的
        LOG_INFO("Created Rt Neutron Index with direct half: %s", index_name_.c_str());
        gpu_index_ =
            new neutron::gpu::NeutronIndexInterface(neutron::gpu::NeutronIndexInterface::CALCULATORTYPE::DIRECTHALF,
                                                    index_meta.dimension(), index_meta.sizeofElement(), device_no);
    } else {
        // 索引是float的, 需要转换成半精度的
        if (FLAGS_global_half_force_control != 2 &&
            ((FLAGS_global_half_force_control == 1) || (index_->GetIndexParams().has(PARAM_VECTOR_ENABLE_HALF) &&
                                                        index_->GetIndexParams().getBool(PARAM_VECTOR_ENABLE_HALF)))) {
            LOG_INFO("Created Rt Neutron Index with half float: %s", index_name_.c_str());
            gpu_index_ = new neutron::gpu::NeutronIndexInterface(
                neutron::gpu::NeutronIndexInterface::CALCULATORTYPE::HALF, index_meta.dimension(),
                index_meta.sizeofElement() / 2, device_no);
        } else {
            LOG_INFO("Created Rt Neutron Index with float: %s", index_name_.c_str());
            gpu_index_ =
                new neutron::gpu::NeutronIndexInterface(neutron::gpu::NeutronIndexInterface::CALCULATORTYPE::FLOAT,
                                                        index_meta.dimension(), index_meta.sizeofElement(), device_no);
        }
    }

    // 通用
    neutron_manager_interface_ = GpuResourcesWrapper::GetInstance()->GetNeutronManagerInterface();
    gpu_index_->SetNeutronManagerInterface(neutron_manager_interface_);
    // 预留实时余量
    auto index_params = index_->GetIndexParams();
    if (index_->GetIndexParams().has(PARAM_GENERAL_MAX_BUILD_NUM)) {
        max_build_num_ = index_->GetIndexParams().getUint32(PARAM_GENERAL_MAX_BUILD_NUM);
    }
    gpu_index_->Reserve(max_build_num_);
    LOG_INFO("Reserve Rt Neutron Index: %s, %u", index_name_.c_str(), max_build_num_);
    is_gpu_rt_ = true;
    index_->SetSearcher(this);
}

bool GpuGroupIvfSearcher::Add(docid_t doc_id, const void *data)
{
    if (!gpu_index_) {
        LOG_ERROR("Rt Neutron Index is nullptr, %s", index_name_.c_str());
        return false;
    }
    if (doc_id >= max_build_num_) {
        LOG_ERROR("Rt Neutron Index exceeds, %s, %u, %u", index_name_.c_str(), doc_id, max_build_num_);
        return false;
    }
    if (doc_id != gpu_index_->GetDocNum()) {
        LOG_ERROR("Rt Neutron Index unmatch, %s, %u, %u", index_name_.c_str(), doc_id, gpu_index_->GetDocNum());
        return false;
    }
    gpu_index_->CopyToGpu(data, index_->GetIndexMeta().sizeofElement());
    return true;
}

void GpuGroupIvfSearcher::SetBaseDocId(exdocid_t baseDocId)
{
    base_docid_ = baseDocId;
    index_->SetBaseDocid(baseDocId);
}

IndexMeta::FeatureTypes GpuGroupIvfSearcher::getFType()
{
    return this->index_->GetIndexMeta().type();
}

//! search by query
int GpuGroupIvfSearcher::Search(const QueryInfo &query_info, GeneralSearchContext *context)
{
    butil::Timer timer;
    timer.start();
    if (query_info.GetDimension() != index_->GetIndexMeta().dimension()) {
        LOG_ERROR("query dimension %lu != index dimension %lu.", query_info.GetDimension(),
                  index_->GetIndexMeta().dimension());
        return -1;
    }

    const std::vector<GroupInfo> &group_infos = query_info.GetGroupInfos();
    uint32_t group_num = group_infos.size();
    if (query_info.MultiQueryMode()) {
        if (query_info.GetVectors().size() != group_num) {
            LOG_ERROR("num of query vector is not equal to group, %lu != %lu", query_info.GetVectors().size(),
                      group_num);
            return false;
        }
    }

    size_t total_topk = query_info.GetTotalRecall();
    if (total_topk <= 0) {
        LOG_ERROR("total_topk size error, %u", total_topk);
        return -1;
    }
    if (total_topk > 1500 && sort_mode_ == 0) {
        LOG_INFO("topk larger than 1500 is not supported in gpu by sort mode 0, topk: %u, rewrite to 1500", total_topk);
        total_topk = 1500;
    }
    if (total_topk > 5000 && sort_mode_ == 1) {
        LOG_INFO("topk larger than 5000 is not supported in gpu by sort mode 1, topk: %u, rewrite to 5000", total_topk);
        total_topk = 5000;
    }

    auto &real_slot_indexs = context->getAllGroupRealSlotIndexs();
    std::vector<CoarseIndex<SmallBlock>::PostingIterator> ivf_postings;
    std::vector<uint32_t> group_doc_nums(group_num, 0);
    std::vector<uint32_t> ivf_postings_group_ids;
    if (CollectIvfPostings(query_info, group_doc_nums, ivf_postings, real_slot_indexs, ivf_postings_group_ids,
                           context) != 0) {
        LOG_ERROR("collect ivf posting failed.");
        return -1;
    }

    uint32_t total_node_count = 0;
    for (uint32_t i = 0; i < group_num; i++) {
        total_node_count += group_doc_nums[i];
    }

    MONITOR_METRIC_LOG(GpuGroupIvf_GroupNum, group_num);
    if (base_docid_ == 0) {
        MONITOR_METRIC_LOG(GpuGroupIvf_FullDocNum, total_node_count);
    } else {
        MONITOR_METRIC_LOG(GpuGroupIvf_RtDocNum, total_node_count);
    }

    if (total_node_count == 0) {
        LOG_WARN("dist_nodes size is 0");
        timer.stop();
        transaction_GpuPreProcessTime(timer.u_elapsed(), true);
        return 0;
    }

    total_topk = std::min<size_t>(total_topk, total_node_count);

    if (index_->MultiAgeMode() && query_info.MultiAgeMode()) {
        neutron::gpu::GpuDataParam gpu_data_param(index_->GetIndexMeta().dimension(), 1,
                                                  index_->GetIndexMeta().sizeofElement() /
                                                      index_->GetIndexMeta().dimension());
        neutron::gpu::GpuDataOffset gpu_data_offset;
        std::vector<neutron::gpu::QueryDataParam> query_data_params;
        query_data_params.emplace_back(total_node_count, total_topk, query_info.GetAgeInfos().size());

        // 目前只接入了单请求的接口，multi query待接入
        neutron::gpu::AddQueryParam(gpu_data_param, query_data_params.at(0));
        neutron::gpu::CalcGpuData(gpu_data_param, gpu_data_offset);

        char *cpu_data = (char *)context->GetSessionPool()->allocateAlign(
            gpu_data_offset.memcpy_data_bytes + gpu_data_offset.result_bytes + gpu_data_offset.distance_bytes, 32);

        std::vector<std::vector<uint32_t>> sort_list_doc_nums;
        if (FillCustomData(cpu_data, gpu_data_param, gpu_data_offset, query_data_params, sort_list_doc_nums,
                           ivf_postings, query_info, context) != 0) {
            LOG_ERROR("FillData in MultiAgeMode failed.");
            return -1;
        }
        FillGeneralData(cpu_data, gpu_data_param, gpu_data_offset, query_data_params, sort_list_doc_nums);

        timer.stop();
        transaction_GpuPreProcessTime(timer.u_elapsed(), true);
        timer.start();

        if (gpu_index_->SearchUnify(cpu_data, (uint32_t *)&gpu_data_param, (uint64_t *)&gpu_data_offset) != 0) {
            LOG_ERROR("Search Unify Failed.");
            return -1;
        }

        // 直接将结果塞入context中
        context->Result().reserve(context->Result().size() + total_topk);
        auto docid_list = (uint32_t *)(cpu_data + gpu_data_offset.docid_list);
        auto gpu_scores = (float *)(cpu_data + gpu_data_offset.cpu_result);
        auto gpu_ids = (uint32_t *)(gpu_scores + total_topk);
        for (size_t i = 0; i < query_info.GetAgeInfos().size(); i++) {
            uint32_t final_age_topk = std::min<uint32_t>(total_topk, query_info.GetAgeInfos().at(i).topk);
            for (size_t k = 0; k < final_age_topk; k++) {
                if (gpu_ids[k] == -1) {
                    break;
                }
                context->emplace_back(0, base_docid_ + docid_list[gpu_ids[k]], gpu_scores[k], i);
            }
            gpu_scores += total_topk * 2;
            gpu_ids += total_topk * 2;
        }

        timer.stop();
        transaction_GpuProcessTime(timer.u_elapsed(), true);

        timer.start();
        PostProcess(context, group_infos.size());
        timer.stop();
        transaction_GpuPostProcessTime(timer.u_elapsed(), true);

        return 0;
    }

    if (query_info.MultiQueryMode()) {
        // 3. 计算所需字节数
        uint32_t task_num = group_num;
        putil::mem_pool::PoolVector<uint64_t> data_offsets(context->GetSessionPool(), 8, 0);
        putil::mem_pool::PoolVector<uint32_t> start_ends(context->GetSessionPool(), task_num * 4, 0);
        uint64_t allocated_bytes = 0, aligned_doc_num = 0, max_topk = 0, sort_block_num = 0, max_len = 0;

        for (uint32_t i = 0; i < task_num; i++) {
            max_len = std::max<uint32_t>(max_len, group_doc_nums[i]);
            start_ends[i * 4] = aligned_doc_num;
            start_ends[i * 4 + 1] = aligned_doc_num + group_doc_nums[i];
            start_ends[i * 4 + 2] = aligned_doc_num + (group_doc_nums[i] + 511) / 512 * 512;
            start_ends[i * 4 + 3] = sort_block_num + (group_doc_nums[i] + 10239) / 10240;
            aligned_doc_num = start_ends[i * 4 + 2];
            sort_block_num = start_ends[i * 4 + 3];
            max_topk = std::max<uint64_t>(max_topk, query_info.GetTopks()[i]);
            // if (max_topk > group_doc_nums[i]) {
            //     LOG_ERROR("search range is too small, topk: %d, doc_nums: %d, task_no: %d, raw query: %s", max_topk,
            //               group_doc_nums[i], i, const_cast<QueryInfo &>(query_info).GetRawQuery().c_str());
            //     return -1;
            // }
            if (max_topk > 1500 && sort_mode_ == 0) {
                LOG_INFO("topk larger than 1500 is not supported in gpu by sort mode 0, topk: %u, rewrite to 1500",
                         max_topk);
                max_topk = 1500;
            }
            if (max_topk > 5000 && sort_mode_ == 1) {
                LOG_INFO("topk larger than 5000 is not supported in gpu by sort mode 1, topk: %u, rewrite to 5000",
                         max_topk);
                max_topk = 5000;
            }
        }

        // 4. 分配内存
        // pq_codebook
        uint64_t query_size = index_->GetIndexMeta().sizeofElement();
        allocated_bytes += task_num * query_size;
        // start end
        data_offsets[0] = allocated_bytes;
        allocated_bytes += task_num * 4 * sizeof(uint32_t);
        // docid
        data_offsets[1] = allocated_bytes;
        allocated_bytes += aligned_doc_num * sizeof(uint32_t);
        // result
        data_offsets[2] = allocated_bytes;
        allocated_bytes += task_num * max_topk * (sizeof(float) + sizeof(uint32_t));

        // 5. 写入真实数据
        char *batch_data = (char *)context->GetSessionPool()->allocateAlign(allocated_bytes, 32);
        // query
        char *write_data = batch_data;
        for (uint32_t i = 0; i < task_num; i++) {
            memcpy(write_data, query_info.GetVectors().at(i), query_size);
            write_data += query_size;
        }
        // start end
        write_data = batch_data + data_offsets[0];
        memcpy(write_data, start_ends.data(), start_ends.size() * sizeof(uint32_t));
        // docid
        write_data = batch_data + data_offsets[1];
        for (uint32_t i = 0, ivf_posting_index = 0; i < task_num; i++) {
            uint32_t *task_write_data = (uint32_t *)write_data + start_ends[i * 4];
            uint32_t real_node_count = 0;
            if (!is_gpu_rt_) {
                for (size_t j = 0; j < real_slot_indexs[i].size(); j++) {
                    uint32_t slot_doc_num = index_->GetFlatCoarseIndex()->GetSlotDocNum(real_slot_indexs[i][j]);
                    uint32_t slot_start_index = index_->GetFlatCoarseIndex()->GetStartIndexs(real_slot_indexs[i][j]);
                    memcpy(task_write_data, index_->GetFlatCoarseIndex()->GetSlotDocIds() + slot_start_index,
                           slot_doc_num * sizeof(uint32_t));
                    task_write_data += slot_doc_num;
                    real_node_count += slot_doc_num;
                }
            } else {
                for (; ivf_posting_index < ivf_postings.size() && i == ivf_postings_group_ids[ivf_posting_index];
                     ivf_posting_index++) {
                    CoarseIndex<SmallBlock>::PostingIterator &iter = ivf_postings.at(ivf_posting_index);
                    while (UNLIKELY(!iter.finish())) {
                        uint32_t docid = iter.next();
                        *task_write_data = docid;
                        real_node_count++;
                        task_write_data++;
                    }
                }
            }

            if (real_node_count != group_doc_nums[i]) {
                LOG_ERROR("total_node_count not equal to total_node_count: %u, %u, %d", real_node_count,
                          group_doc_nums[i], i);
                return -1;
            }

            memset((uint32_t *)write_data + start_ends[i * 4 + 1], 0,
                   (start_ends[i * 4 + 2] - start_ends[i * 4 + 1]) * sizeof(uint32_t));
        }

        timer.stop();
        transaction_GpuPreProcessTime(timer.u_elapsed(), true);
        timer.start();

        // 6. 请求gpu
        if (gpu_index_->SearchBatch(task_num, sort_block_num, max_topk, max_len, batch_data, data_offsets.data(),
                                    allocated_bytes) != 0) {
            LOG_ERROR("gpu_index Search Batch failed");
            timer.stop();
            transaction_GpuProcessTime(timer.u_elapsed(), false);
            return -1;
        }

        timer.stop();
        transaction_GpuProcessTime(timer.u_elapsed(), true);
        timer.start();

        // 7. 直接将结果塞入context中
        write_data = batch_data + data_offsets[2];
        float *out = (float *)write_data;
        uint32_t *out_idx = (uint32_t *)write_data + max_topk * (sort_mode_ ? task_num : 1);
        context->Result().reserve(context->Result().size() + max_topk * task_num);
        for (size_t i = 0; i < task_num; i++) {
            for (size_t j = 0; j < std::min<uint64_t>(max_topk, group_doc_nums[i]); j++) {
                if (sort_mode_ && (out_idx[j] < 0 || out_idx[j] >= index_->GetDocNum())) {
                    LOG_ERROR("final result failed, %s, %d, %u, %d, %s", index_name_.c_str(), sort_mode_, j, out_idx[j],
                              const_cast<QueryInfo &>(query_info).GetRawQuery().c_str());
                    return -1;
                }
                if (!sort_mode_ && (out_idx[j] < 0 || out_idx[j] >= group_doc_nums[i])) {
                    LOG_ERROR("final result failed, %s, %d, %u, %d, %s", index_name_.c_str(), sort_mode_, j, out_idx[j],
                              const_cast<QueryInfo &>(query_info).GetRawQuery().c_str());
                    return -1;
                }
                auto dist_node_data = (uint32_t *)(batch_data + data_offsets[1] + start_ends[i * 4] * sizeof(uint32_t));
                context->emplace_back(0, base_docid_ + (sort_mode_ ? out_idx[j] : dist_node_data[out_idx[j]]), out[j],
                                      i);
            }
            out += sort_mode_ ? max_topk : max_topk * 2;
            out_idx += sort_mode_ ? max_topk : max_topk * 2;
        }
    } else {
        std::vector<uint32_t> dist_nodes;
        if (CollectIvfDistNodes(query_info, dist_nodes, ivf_postings, context) != 0) {
            LOG_ERROR("collect ivf dist nodes failed.");
            return -1;
        }

        timer.stop();
        transaction_GpuPreProcessTime(timer.u_elapsed(), true);

        timer.start();
        if (enable_batch_ && dist_nodes.size() < FLAGS_batch_scheduler_doc_num_threshold) {
            char *result_data =
                (char *)context->GetSessionPool()->allocateAlign(total_topk * (sizeof(float) + sizeof(uint32_t)), 32);
            bthread_cond_t batch_cond;
            bthread_mutex_t batch_mutex;
            bthread_cond_init(&batch_cond, NULL);
            bthread_mutex_init(&batch_mutex, NULL);

            std::shared_ptr<GpuGroupIvfBatchTask> batch_task(new GpuGroupIvfBatchTask(
                &dist_nodes, &query_info, total_topk, group_num, &group_doc_nums, &result_data, false,
                /*!!in_mem_searcher_*/ false, context->GetSessionPool(), &batch_cond, &batch_mutex));

            bthread_mutex_lock(&batch_mutex);
            size_t scheduler_id = random_() & (FLAGS_batch_scheduler_num - 1);
            if (basic_batch_schedulers_[scheduler_id]->Schedule(batch_task) != true) {
                LOG_ERROR("Schedule Batch failed");
                return -1;
            }

            int retry_num = 5; // 重试5次
            while (!batch_task->is_finish && retry_num--) {
                bthread_cond_wait(&batch_cond, &batch_mutex);
                if (retry_num < 4) {
                    LOG_ERROR("spurious waked up, %p, %p, %p, %d, %d, %d", &batch_cond, &batch_mutex, batch_cond.seq,
                              *(batch_cond.seq), errno, retry_num);
                }
            }

            bthread_mutex_unlock(&batch_mutex);

            if (!batch_task->is_finish) { // 被虚假或异常唤醒的，重试仍然有问题
                LOG_ERROR("Batch Task is not finished, %p, %p, %p, %d, %d", &batch_cond, &batch_mutex, batch_cond.seq,
                          *(batch_cond.seq), errno);
                bthread_mutex_destroy(&batch_mutex);
                bthread_cond_destroy(&batch_cond);
                batch_task->is_destroy = true;
                return -1;
            }

            if (!batch_task->is_success) { // GPU Batch处理有问题的（没拿到资源/Topk过大）
                LOG_ERROR("Batch Task is not successful, %p, %p, %p, %d, %d", &batch_cond, &batch_mutex, batch_cond.seq,
                          *(batch_cond.seq), errno);
                bthread_mutex_destroy(&batch_mutex);
                bthread_cond_destroy(&batch_cond);
                batch_task->is_destroy = true;
                return -1;
            }

            bthread_mutex_destroy(&batch_mutex);
            bthread_cond_destroy(&batch_cond);

            batch_task->is_destroy = true;

            float *distances = (float *)result_data;
            uint32_t *labels = (uint32_t *)distances + total_topk;
            context->Result().reserve(context->Result().size() + total_topk);
            for (size_t i = 0; i < total_topk; i++) {
                context->emplace_back(0, base_docid_ + (sort_mode_ ? labels[i] : dist_nodes[labels[i]]), distances[i]);
            }
        } else {
            std::vector<float> distances(total_topk);
            std::vector<int> labels(total_topk);
            if (gpu_index_->Search(query_info.GetVector(), distances, labels, dist_nodes, 1 /*qnum*/, total_topk,
                                   nullptr, 0) != 0) {
                LOG_ERROR("gpu_index Search failed");
                return -1;
            }

            // 直接将结果塞入context中
            context->Result().reserve(context->Result().size() + distances.size());
            for (size_t i = 0; i < distances.size(); i++) {
                context->emplace_back(0, base_docid_ + (sort_mode_ ? labels[i] : dist_nodes[labels[i]]), distances[i]);
            }
        }
        timer.stop();
        transaction_GpuProcessTime(timer.u_elapsed(), true);

        timer.start();
    }

    PostProcess(context, group_infos.size());
    timer.stop();
    transaction_GpuPostProcessTime(timer.u_elapsed(), true);

    return 0;
}

int GpuGroupIvfSearcher::CollectIvfPostings(const QueryInfo &query_info, std::vector<uint32_t> &group_doc_nums,
                                            std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings,
                                            std::vector<std::vector<off_t>> &real_slot_indexs,
                                            std::vector<uint32_t> &ivf_postings_group_ids,
                                            GeneralSearchContext *context)
{
    const std::vector<GroupInfo> &group_infos = query_info.GetGroupInfos();
    const size_t group_nums = group_infos.size();

    bool is_rt = !real_slot_indexs.empty() && is_gpu_rt_;
    bool is_multi_age = query_info.MultiAgeMode();
    bool is_recall_mode = query_info.GetContextParams().has(PARAM_GENERAL_RECALL_TEST_MODE);
    bool need_truncate = (is_rt && !is_recall_mode);

    if (is_rt && is_multi_age && index_->GetIndexParams().has(PARAM_RT_COARSE_SCAN_RATIO)) {
        real_slot_indexs.clear();
    }

    if (real_slot_indexs.empty()) {
        real_slot_indexs.resize(group_nums);
        for (size_t i = 0; i < group_nums; i++) {
            gindex_t group_index = index_->GetGroupManager().GetGroupIndex(group_infos.at(i));
            if (group_index == INVALID_GROUP_INDEX) {
                LOG_WARN("group not in group manager. level:%d, id:%d", group_infos.at(i).level, group_infos.at(i).id);
                continue;
            }

            if (index_->SearchIvf(group_index,
                                  query_info.MultiQueryMode() ? query_info.GetVectors().at(i)
                                                              : query_info.GetVectors().at(0),
                                  query_info.GetVectorLen(), query_info.GetDimension(), query_info.GetContextParams(),
                                  real_slot_indexs[i], is_rt, is_multi_age) != 0) {
                LOG_ERROR("Failed to call SearchIvf.");
                return -1;
            }
        }
    }
    index_->RecoverPostingFromSlot(ivf_postings, ivf_postings_group_ids, real_slot_indexs, group_doc_nums,
                                   need_truncate, query_info.MultiQueryMode());

    return 0;
}

int GpuGroupIvfSearcher::CollectIvfDistNodes(const QueryInfo &query_info, std::vector<uint32_t> &dist_nodes,
                                             std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings,
                                             GeneralSearchContext *context)
{
    const std::vector<GroupInfo> &group_infos = query_info.GetGroupInfos();
    const size_t group_nums = group_infos.size();
    auto &real_slot_indexs = context->getAllGroupRealSlotIndexs();
    size_t node_count = 0;
    for (size_t i = 0; i < ivf_postings.size(); i++) {
        node_count += ivf_postings.at(i).getDocNum();
    }
    if (node_count == 0) {
        LOG_WARN("node_count size is zero");
        return 0;
    }
    dist_nodes.resize(node_count);

    size_t record_centroid_num = 0;
    if (base_docid_ == 0) {
        auto flat_coarse_index = index_->GetFlatCoarseIndex();
        uint32_t *dist_nodes_data = dist_nodes.data();
        uint32_t *slot_doc_ids = flat_coarse_index->GetSlotDocIds();
        for (size_t i = 0; i < group_nums; i++) {
            for (size_t j = 0; j < real_slot_indexs[i].size(); j++) {
                uint32_t real_slot_index = real_slot_indexs[i][j];
                uint32_t slot_doc_num = flat_coarse_index->GetSlotDocNum(real_slot_index);
                uint32_t slot_start_index = flat_coarse_index->GetStartIndexs(real_slot_index);
                memcpy(dist_nodes_data, slot_doc_ids + slot_start_index, slot_doc_num * sizeof(uint32_t));
                dist_nodes_data += slot_doc_num;
                record_centroid_num++;
            }
        }
    } else {
        size_t node_index = 0;
        for (size_t i = 0; i < ivf_postings.size(); i++) {
            CoarseIndex<SmallBlock>::PostingIterator &iter = ivf_postings.at(i);
            while (UNLIKELY(!iter.finish())) {
                uint32_t docid = iter.next();
                dist_nodes.at(node_index++) = docid;
            }
            record_centroid_num++;
        }
    }

    MONITOR_METRIC_LOG(GpuGroupIvf_CentroidNum, record_centroid_num);

    return 0;
}

// 目前只适用于MultiAgeMode
int GpuGroupIvfSearcher::FillCustomData(char *cpu_data, neutron::gpu::GpuDataParam &gpu_data_param,
                                        neutron::gpu::GpuDataOffset &gpu_data_offset,
                                        std::vector<neutron::gpu::QueryDataParam> &query_data_params,
                                        std::vector<std::vector<uint32_t>> &sort_list_doc_nums,
                                        std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings,
                                        const QueryInfo &query_info, GeneralSearchContext *context)
{
    const std::vector<AgeInfo> &age_infos = query_info.GetAgeInfos();
    std::vector<uint32_t> age_counters(age_infos.size(), 0);
    int64_t now_timestamp_s = butil::gettimeofday_s();

    const std::vector<GroupInfo> &group_infos = query_info.GetGroupInfos();
    const size_t group_nums = group_infos.size();
    auto &real_slot_indexs = context->getAllGroupRealSlotIndexs();
    auto flat_coarse_index = index_->GetFlatCoarseIndex();
    uint32_t *dist_nodes_data = (uint32_t *)(cpu_data + gpu_data_offset.docid_list);
    uint32_t *slot_doc_ids = flat_coarse_index->GetSlotDocIds();
    size_t record_centroid_num = 0;

    uint32_t *age_data = (uint32_t *)(cpu_data + gpu_data_offset.cpu_result + gpu_data_offset.result_bytes);

    // 1. id和age赋值
    if (!is_gpu_rt_) {
        for (size_t i = 0; i < group_nums; i++) {
            for (size_t j = 0; j < real_slot_indexs[i].size(); j++) {
                uint32_t real_slot_index = real_slot_indexs[i][j];
                uint32_t slot_doc_num = flat_coarse_index->GetSlotDocNum(real_slot_index);
                uint32_t slot_start_index = flat_coarse_index->GetStartIndexs(real_slot_index);
                memcpy(dist_nodes_data, slot_doc_ids + slot_start_index, slot_doc_num * sizeof(uint32_t));
                for (uint32_t k = 0; k < slot_doc_num; k++) {
                    uint32_t create_timestamp_s =
                        *(uint32_t *)index_->GetDocCreateTimeProfile().getInfo(dist_nodes_data[k]);
                    uint32_t doc_age = now_timestamp_s - create_timestamp_s;
                    size_t m = 0;
                    for (; m < age_infos.size(); m++) {
                        if (doc_age <= age_infos[m].age) {
                            age_counters[m]++;
                            age_data[k] = m;
                            break;
                        }
                    }
                    if (m == age_infos.size()) {
                        // 非法, 比30day还大
                        age_data[k] = -1;
                    }
                }
                dist_nodes_data += slot_doc_num;
                age_data += slot_doc_num;
                record_centroid_num++;
            }
        }
    } else {
        for (size_t i = 0; i < ivf_postings.size(); i++) {
            uint32_t slot_doc_num = 0;
            CoarseIndex<SmallBlock>::PostingIterator &iter = ivf_postings.at(i);
            while (UNLIKELY(!iter.finish())) {
                uint32_t docid = iter.next();
                *dist_nodes_data = docid;
                uint32_t create_timestamp_s = *(uint32_t *)index_->GetDocCreateTimeProfile().getInfo(docid);
                uint32_t doc_age = now_timestamp_s - create_timestamp_s;
                size_t m = 0;
                for (; m < age_infos.size(); m++) {
                    if (doc_age <= age_infos[m].age) {
                        age_counters[m]++;
                        age_data[slot_doc_num] = m;
                        break;
                    }
                }
                if (m == age_infos.size()) {
                    // 非法, 比30day还大
                    age_data[slot_doc_num] = -1;
                }
                slot_doc_num++;
                dist_nodes_data++;
            }
            age_data += slot_doc_num;
            record_centroid_num++;
        }
    }

    // TODO: multi query
    memset(dist_nodes_data, 0,
           (query_data_params[0].aligned_doc_num - query_data_params[0].doc_num) * sizeof(uint32_t));

    MONITOR_METRIC_LOG(GpuGroupIvf_CentroidNum, record_centroid_num);

    // 2. 赋值query vec 和 sort id list
    sort_list_doc_nums.push_back(age_counters);
    char *write_data = cpu_data + gpu_data_offset.query_vec;
    for (uint32_t i = 0; i < gpu_data_param.query_num; i++) {
        memcpy(write_data, query_info.GetVectors().at(i), index_->GetIndexMeta().sizeofElement());
        write_data += index_->GetIndexMeta().sizeofElement();
    }

    uint32_t *sort_docid_list = (uint32_t *)(cpu_data + gpu_data_offset.sort_docid_list);
    age_data = (uint32_t *)(cpu_data + gpu_data_offset.cpu_result + gpu_data_offset.result_bytes);
    std::vector<uint32_t> age_cursors(age_infos.size() + 1, 0);
    for (size_t i = 1; i < age_cursors.size(); ++i) {
        age_cursors[i] = age_cursors[i - 1] + age_counters[i - 1];
    }
    for (uint32_t i = 0; i < gpu_data_param.query_num; i++) {
        for (uint32_t j = 0; j < query_data_params[i].doc_num; j++) {
            if (age_data[j] == -1) {
                continue;
            }
            sort_docid_list[age_cursors[age_data[j]]++] = j;
        }
    }

    return 0;
}

int GpuGroupIvfSearcher::PostProcess(GeneralSearchContext *context, size_t group_num) const
{
    std::vector<SearchResult> &results = context->Result();
    std::sort(results.begin(), results.end(), [](const SearchResult &a, const SearchResult &b) {
        if (a.gloid == b.gloid) {
            return a.score < b.score;
        }
        return a.gloid < b.gloid;
    });
    // 如果多于一个分组，返回结果去重
    if (group_num > 1) {
        results.erase(std::unique(results.begin(), results.end(),
                                  [](const SearchResult &a, const SearchResult &b) { return a.gloid == b.gloid; }),
                      results.end());
    }

    MONITOR_METRIC_LOG(GpuGroupIvf_FullResultDocNum, results.size());

    return 0;
}


void GpuGroupIvfSearcher::PushSearchResultToContext(GeneralSearchContext *context, std::vector<uint32_t> &dist_nodes,
                                                    std::vector<float> &distances, std::vector<int> &labels) const
{
    bool with_pk = index_->WithPk();
    context->Result().reserve(context->Result().size() + distances.size());
    for (size_t i = 0; i < distances.size(); i++) {
        pk_t pk = 0;
        if (unlikely(with_pk)) {
            pk = index_->GetPk(dist_nodes[labels[i]]);
        }

        context->emplace_back(pk, dist_nodes[labels[i]], distances[i]);
    }
}

MERCURY_NAMESPACE_END(core);
#endif // ENABLE_GPU_IN_MERCURY_