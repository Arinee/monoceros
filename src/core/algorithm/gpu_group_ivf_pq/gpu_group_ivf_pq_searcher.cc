#ifdef ENABLE_GPU_IN_MERCURY_
#include "src/core/algorithm/gpu_group_ivf_pq/gpu_group_ivf_pq_searcher.h"
#include "bthread/bthread.h"
#include "putil/mem_pool/PoolVector.h"
#include "src/core/algorithm/thread_common.h"
#include "src/core/framework/utility/closure.h"

DECLARE_uint32(global_enable_device_count);

MERCURY_NAMESPACE_BEGIN(core);

GpuGroupIvfPqSearcher::GpuGroupIvfPqSearcher()
{
    if (!index_) {
        index_.reset(new GroupIvfPqIndex());
    }
    SetThreadEnv();
}

GpuGroupIvfPqSearcher::~GpuGroupIvfPqSearcher()
{
    if (own_gpu_index_ && gpu_index_) {
        delete gpu_index_;
        gpu_index_ = nullptr;
    }
}

int GpuGroupIvfPqSearcher::Init(IndexParams &params)
{
    index_->SetIndexParams(params);
    index_->SetIsGpu(true);

    std::string index_name = params.getString(PARAM_VECTOR_INDEX_NAME);
    LOG_INFO("Start Init GpuGroupIvfPqSearcher, %s", index_name.c_str());

    MONITOR_TRANSACTION(GpuGroupIvfPq, GpuProcessTime);
    MONITOR_TRANSACTION(GpuGroupIvfPq, GpuPreProcessTime);
    MONITOR_TRANSACTION(GpuGroupIvfPq, GpuPostProcessTime);
    MONITOR_METRIC_WITH_INDEX(GpuGroupIvfPq_GroupNum, "GpuGroupIvfPq_GroupNum_" + index_name);
    MONITOR_METRIC_WITH_INDEX(GpuGroupIvfPq_FullDocNum, "GpuGroupIvfPq_FullDocNum_" + index_name);
    MONITOR_METRIC_WITH_INDEX(GpuGroupIvfPq_RtDocNum, "GpuGroupIvfPq_RtDocNum_" + index_name);
    index_name_ = index_name;
    char *pEnd;
    sort_mode_ = std::strtol(std::getenv("PARAM_SORT_MODE"), &pEnd, 10);

    LOG_INFO("End Init GpuGroupIvfSearcher, %s", index_name.c_str());

    return 0;
}

int GpuGroupIvfPqSearcher::LoadIndex(const std::string &path)
{
    // TODO
    return -1;
}

int GpuGroupIvfPqSearcher::LoadIndex(const void *data, size_t size)
{
    if (index_->Load(data, size) != 0) {
        LOG_ERROR("Failed to load index.");
        return -1;
    }
    dist_scorer_factory_.Init(index_->GetIndexMeta());

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

    LOG_INFO("Created Neutron Index with pq: %s", index_name.c_str());
    gpu_index_ = new neutron::gpu::NeutronIndexInterface(neutron::gpu::NeutronIndexInterface::CALCULATORTYPE::PQ, 32,
                                                         32 * sizeof(uint16_t) /* TODO:优化成uint8*/, device_no);
    neutron_manager_interface_ = GpuResourcesWrapper::GetInstance()->GetNeutronManagerInterface();
    gpu_index_->SetNeutronManagerInterface(neutron_manager_interface_);

    // 将索引中的向量全部拷贝到gpu device中
    gpu_index_->Add((const char *)index_->GetPqCodeProfile().getInfo(0), index_->GetDocNum());
    std::cout << "load num: " << index_->GetDocNum() << std::endl;
    own_gpu_index_ = true;
    return 0;
}

void GpuGroupIvfPqSearcher::SetIndex(Index::Pointer index)
{
    index_ = std::dynamic_pointer_cast<GroupIvfPqIndex>(index);
    dist_scorer_factory_.Init(index_->GetIndexMeta());
}

void GpuGroupIvfPqSearcher::SetBaseDocId(exdocid_t baseDocId)
{
    base_docid_ = baseDocId;
    index_->SetBaseDocid(baseDocId);
}

IndexMeta::FeatureTypes GpuGroupIvfPqSearcher::getFType()
{
    return this->index_->GetIndexMeta().type();
}

//! search by query
int GpuGroupIvfPqSearcher::Search(const QueryInfo &query_info, GeneralSearchContext *context)
{
    if (index_->MultiAgeMode() && query_info.MultiAgeMode()) {
        LOG_ERROR("GpuGroupIvfPq doesn't support multi age mode.");
        return -1;
    }

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
    // 定时器
    butil::Timer timer;
    timer.start();

    // 1. 取出最近的scan_ratio个中心点
    std::vector<std::vector<off_t>> &real_slot_indexs = context->getAllGroupRealSlotIndexs();
    std::vector<uint32_t> group_doc_nums(group_infos.size(), 0);
    std::vector<CoarseIndex<BigBlock>::PostingIterator> ivf_postings;
    std::vector<uint32_t> ivf_postings_group_ids;
    if (CollectIvfPostings(ivf_postings, ivf_postings_group_ids, group_doc_nums, query_info, real_slot_indexs) != 0) {
        LOG_ERROR("collect ivf posting failed.");
        return -1;
    }

    // 2. 统计计算doc数量
    uint32_t total_node_count = 0;
    for (uint32_t i = 0; i < ivf_postings.size(); i++) {
        total_node_count += ivf_postings[i].getDocNum();
    }

    MONITOR_METRIC_LOG(GpuGroupIvfPq_GroupNum, group_num);
    if (base_docid_ == 0) {
        MONITOR_METRIC_LOG(GpuGroupIvfPq_FullDocNum, total_node_count);
    } else {
        MONITOR_METRIC_LOG(GpuGroupIvfPq_RtDocNum, total_node_count);
    }

    if (total_node_count == 0 || total_node_count >= 2000000) {
        LOG_WARN("dist_nodes size error: %u", total_node_count);
        timer.stop();
        transaction_GpuPreProcessTime(timer.u_elapsed(), false);
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
            if (max_topk > group_doc_nums[i]) {
                LOG_ERROR("search range is too small, topk: %d, doc_nums: %d, task_no: %d, raw query: %s", max_topk,
                          group_doc_nums[i], i, const_cast<QueryInfo &>(query_info).GetRawQuery().c_str());
                return -1;
            }
            if (max_topk > 1500 && sort_mode_ == 0) {
                LOG_INFO("topk larger than 1500 is not supported in gpu by sort mode 0, topk: %u, rewrite to 1500", max_topk);
                max_topk = 1500;
            }
            if (max_topk > 5000 && sort_mode_ == 1) {
                LOG_INFO("topk larger than 5000 is not supported in gpu by sort mode 1, topk: %u, rewrite to 5000", max_topk);
                max_topk = 5000;
            }
        }

        // 4. 分配内存
        // pq_codebook
        uint64_t query_size = 32 * 256 * sizeof(float);
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
            QueryDistanceMatrix qdm(index_->GetIndexMeta(), &(index_->GetPqCentroidResource()));
            if (!qdm.initDistanceMatrix(query_info.GetVectors().at(i))) {
                LOG_ERROR("Init qdm distance matrix error");
                return -1;
            }
            memcpy(write_data, qdm.GetDistanceArray(), query_size);
            write_data += query_size;
        }
        // start end
        write_data = batch_data + data_offsets[0];
        memcpy(write_data, start_ends.data(), start_ends.size() * sizeof(uint32_t));
        // docid
        write_data = batch_data + data_offsets[1];
        for (uint32_t i = 0; i < task_num; i++) {
            uint32_t *task_write_data = (uint32_t *)write_data + start_ends[i * 4];
            uint32_t real_node_count = 0;
            for (size_t j = 0; j < real_slot_indexs[i].size(); j++) {
                uint32_t slot_doc_num = index_->GetFlatCoarseIndex()->GetSlotDocNum(real_slot_indexs[i][j]);
                uint32_t slot_start_index = index_->GetFlatCoarseIndex()->GetStartIndexs(real_slot_indexs[i][j]);
                memcpy(task_write_data, index_->GetFlatCoarseIndex()->GetSlotDocIds() + slot_start_index,
                       slot_doc_num * sizeof(uint32_t));
                task_write_data += slot_doc_num;
                real_node_count += slot_doc_num;
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
            for (size_t j = 0; j < max_topk; j++) {
                if (sort_mode_ && (out_idx[j] < 0 || out_idx[j] >= index_->GetDocNum())) {
                    LOG_ERROR("final result failed, %s, %d, %u, %d, %s", index_name_.c_str(), sort_mode_, j, out_idx[j],
                              const_cast<QueryInfo &>(query_info).GetRawQuery().c_str());
                    return -1;
                }
                auto dist_node_data = (uint32_t *)(batch_data + data_offsets[1] + start_ends[i * 4] * sizeof(uint32_t));
                context->emplace_back(0, sort_mode_ ? out_idx[j] : dist_node_data[out_idx[j]], out[j], i);
            }
            out += sort_mode_ ? max_topk : max_topk * 2;
            out_idx += sort_mode_ ? max_topk : max_topk * 2;
        }
    } else {
        // 3. 取出倒排链下的点
        total_topk = std::min<size_t>(total_topk, total_node_count);
        std::vector<uint32_t> dist_nodes;
        if (CollectIvfDistNodes(query_info, dist_nodes, ivf_postings, total_node_count, context) != 0) {
            LOG_ERROR("collect ivf dist nodes failed.");
            return -1;
        }

        // 检查
        for (uint32_t i = 0; i < dist_nodes.size(); i++) {
            if (dist_nodes[i] < 0 || dist_nodes[i] >= index_->GetDocNum()) {
                LOG_ERROR("dist nodes error: %u, max: %u", dist_nodes[i], index_->GetDocNum());
                return -1;
            }
        }

        // 4. 初始化码本
        QueryDistanceMatrix qdm(index_->GetIndexMeta(), &(index_->GetPqCentroidResource()));
        if (!qdm.initDistanceMatrix(query_info.GetVector())) {
            LOG_ERROR("Init qdm distance matrix error");
            return -1;
        }

        // 检查
        float *codebook = (float *)qdm.GetDistanceArray();
        for (uint32_t i = 0; i < 32 * 256; i++) {
            if (std::isnan(codebook[i])) {
                LOG_ERROR("codebook is nan");
                return -1;
            }
            if (codebook[i] < -1000.0 || codebook[i] > 1000.0) {
                LOG_ERROR("codebook is too large");
                return -1;
            }
        }

        timer.stop();
        transaction_GpuPreProcessTime(timer.u_elapsed(), true);
        timer.start();

        // 5. 发送GPU计算和排序
        std::vector<float> distances(total_topk);
        std::vector<int> labels(total_topk);
        if (gpu_index_->Search(nullptr, distances, labels, dist_nodes, 1 /*qnum*/, total_topk, qdm.GetDistanceArray(),
                               32 * 256) != 0) {
            LOG_ERROR("gpu_index Search failed");
            timer.stop();
            transaction_GpuProcessTime(timer.u_elapsed(), false);
            return -1;
        }

        timer.stop();
        transaction_GpuProcessTime(timer.u_elapsed(), true);
        timer.start();

        // 6. 直接将结果塞入context中
        context->Result().reserve(context->Result().size() + distances.size());
        for (size_t i = 0; i < distances.size(); i++) {
            if (sort_mode_ && (labels[i] < 0 || labels[i] >= index_->GetDocNum())) {
                LOG_ERROR("final result failed, %s, %d, %u, %d, %s", index_name_.c_str(), sort_mode_, i, labels[i],
                          const_cast<QueryInfo &>(query_info).GetRawQuery().c_str());
                return -1;
            }
            context->emplace_back(0, sort_mode_ ? labels[i] : dist_nodes[labels[i]], distances[i]);
        }
    }

    if (PostProcess(query_info, context, group_infos.size()) != 0) {
        LOG_ERROR("post process result failed.");
        timer.stop();
        transaction_GpuPostProcessTime(timer.u_elapsed(), false);
        return -1;
    }

    timer.stop();
    transaction_GpuPostProcessTime(timer.u_elapsed(), true);

    return 0;
}

int GpuGroupIvfPqSearcher::CollectIvfPostings(std::vector<CoarseIndex<BigBlock>::PostingIterator> &ivf_postings,
                                              std::vector<uint32_t> &ivf_postings_group_ids,
                                              std::vector<uint32_t> &group_doc_nums, const QueryInfo &query_info,
                                              std::vector<std::vector<off_t>> &real_slot_indexs)
{
    const std::vector<GroupInfo> &group_infos = query_info.GetGroupInfos();
    bool is_rt = !real_slot_indexs.empty();
    bool is_multi_age = query_info.MultiAgeMode();
    bool is_recall_mode = query_info.GetContextParams().has(PARAM_GENERAL_RECALL_TEST_MODE);

    if (is_rt && is_multi_age && index_->GetIndexParams().has(PARAM_RT_COARSE_SCAN_RATIO)) {
        real_slot_indexs.clear();
    }

    if (real_slot_indexs.empty()) {
        real_slot_indexs.resize(group_infos.size());
        for (size_t i = 0; i < group_infos.size(); i++) {
            gindex_t group_index = index_->GetGroupManager().GetGroupIndex(group_infos.at(i));
            if (group_index == INVALID_GROUP_INDEX) {
                LOG_WARN("group not in group manager. level:%d, id:%d", group_infos.at(i).level, group_infos.at(i).id);
                continue;
            }

            std::vector<std::pair<off_t, off_t>> level_indexs;
            if (index_->SearchIvf(group_index,
                                  query_info.MultiQueryMode() ? query_info.GetVectors().at(i)
                                                              : query_info.GetVectors().at(0),
                                  query_info.GetVectorLen(), query_info.GetDimension(), query_info.GetContextParams(),
                                  real_slot_indexs[i], level_indexs) != 0) {
                LOG_ERROR("Failed to call SearchIvf.");
                return -1;
            }
        }
    }
    RecoverPostingFromSlot(ivf_postings, ivf_postings_group_ids, real_slot_indexs, group_doc_nums, false,
                           query_info.MultiQueryMode());
    return 0;
}

void GpuGroupIvfPqSearcher::RecoverPostingFromSlot(std::vector<CoarseIndex<BigBlock>::PostingIterator> &postings,
                                                   std::vector<uint32_t> &ivf_postings_group_ids,
                                                   std::vector<std::vector<off_t>> &real_slot_indexs,
                                                   std::vector<uint32_t> &group_doc_nums, bool need_truncate,
                                                   bool is_multi_query)
{
    const CoarseIndex<BigBlock> &coarse_index = index_->GetCoarseIndex();
    for (size_t i = 0; i < real_slot_indexs.size(); ++i) {
        postings.reserve(postings.size() + real_slot_indexs[i].size());
        uint32_t posting_num = 0;
        for (auto &group_real_slot_index : real_slot_indexs[i]) {
            CoarseIndex<BigBlock>::PostingIterator posting = coarse_index.search(group_real_slot_index);
            posting_num += posting.getDocNum();
            group_doc_nums[i] += posting.getDocNum();
            postings.push_back(posting);
            ivf_postings_group_ids.push_back(is_multi_query ? i : 0);
        }
    }
}

int GpuGroupIvfPqSearcher::CollectIvfDistNodes(const QueryInfo &query_info, std::vector<uint32_t> &dist_nodes,
                                               std::vector<CoarseIndex<BigBlock>::PostingIterator> &ivf_postings,
                                               uint32_t total_node_count, GeneralSearchContext *context)
{
    dist_nodes.resize(total_node_count);
    const std::vector<GroupInfo> &group_infos = query_info.GetGroupInfos();
    const size_t group_nums = group_infos.size();
    auto &real_slot_indexs = context->getAllGroupRealSlotIndexs();

    if (base_docid_ == 0) {
        uint32_t real_node_count = 0;
        auto flat_coarse_index = index_->GetFlatCoarseIndex();
        uint32_t *dist_nodes_data = dist_nodes.data();
        uint32_t *slot_doc_ids = flat_coarse_index->GetSlotDocIds();
        for (size_t i = 0; i < group_nums; i++) {
            for (size_t j = 0; j < real_slot_indexs[i].size(); j++) {
                uint32_t real_slot_index = real_slot_indexs[i][j];
                uint32_t slot_doc_num = flat_coarse_index->GetSlotDocNum(real_slot_index);
                real_node_count += slot_doc_num;
                uint32_t slot_start_index = flat_coarse_index->GetStartIndexs(real_slot_index);
                memcpy(dist_nodes_data, slot_doc_ids + slot_start_index, slot_doc_num * sizeof(uint32_t));
                dist_nodes_data += slot_doc_num;
            }
            if (real_node_count != total_node_count) {
                LOG_ERROR("total_node_count not equal to total_node_count: %u, %u", real_node_count, total_node_count)
                return -1;
            }
        }
    } else {
        size_t node_index = 0;
        for (size_t i = 0; i < ivf_postings.size(); i++) {
            CoarseIndex<BigBlock>::PostingIterator &iter = ivf_postings.at(i);
            while (UNLIKELY(!iter.finish())) {
                uint32_t docid = iter.next();
                dist_nodes.at(node_index++) = docid;
            }
        }
    }

    return 0;
}

int GpuGroupIvfPqSearcher::PostProcess(const QueryInfo &query_info, GeneralSearchContext *context,
                                       size_t group_num) const
{
    std::vector<SearchResult> &results = context->Result();

    // calculate real similarity score if doc features are saved
    if (index_->ContainFeature() || vector_retriever_.isValid()) {
        OrigDistScorer scorer = dist_scorer_factory_.Create();
        for (auto &result : results) {
            const void *feature = nullptr;
            if (vector_retriever_.isValid()) {
                // base docid has been added when push to results
                if (!vector_retriever_(result.gloid, feature)) {
                    LOG_ERROR("retrieve vector failed. docid:%lu", result.gloid);
                    continue;
                }
            } else {
                feature = index_->GetFeatureProfile().getInfo(result.gloid - base_docid_);
            }
            if (feature == nullptr) {
                continue;
            }

            float dist = 0.0;
            if (likely(query_info.GetVectors().size() == 1)) {
                dist = scorer.Score(feature, query_info.GetVector());
            } else {
                dist = scorer.Score(feature, query_info.GetVectors().at(result.poolId));
            }
            result.score = dist;
        }
    }

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

    return 0;
}

MERCURY_NAMESPACE_END(core);
#endif // ENABLE_GPU_IN_MERCURY_