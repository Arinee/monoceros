#ifdef ENABLE_GPU_IN_MERCURY_
#include "src/core/algorithm/gpu_ivf_rpq/gpu_ivf_rpq_searcher.h"
#include "bthread/bthread.h"
#include "putil/mem_pool/PoolVector.h"
#include "src/core/algorithm/thread_common.h"
#include "src/core/framework/utility/closure.h"

DECLARE_uint32(global_enable_device_count);

MERCURY_NAMESPACE_BEGIN(core);

GpuIvfRpqSearcher::GpuIvfRpqSearcher()
{
    if (!index_) {
        index_.reset(new IvfRpqIndex());
    }
    SetThreadEnv();
}

GpuIvfRpqSearcher::~GpuIvfRpqSearcher()
{
    if (own_gpu_index_ && gpu_index_) {
        delete gpu_index_;
        gpu_index_ = nullptr;
    }
}

int GpuIvfRpqSearcher::Init(IndexParams &params)
{
    index_->SetIndexParams(params);
    index_->SetIsGpu(true);

    std::string index_name = params.getString(PARAM_VECTOR_INDEX_NAME);
    LOG_INFO("Start Init GpuIvfRpqSearcher, %s", index_name.c_str());

    MONITOR_TRANSACTION(GpuIvfRpq, GpuProcessTime);
    MONITOR_TRANSACTION(GpuIvfRpq, GpuPreProcessTime);
    MONITOR_TRANSACTION(GpuIvfRpq, GpuPostProcessTime);
    MONITOR_METRIC_WITH_INDEX(GpuIvfRpq_GroupNum, "GpuIvfRpq_GroupNum_" + index_name);
    MONITOR_METRIC_WITH_INDEX(GpuIvfRpq_FullDocNum, "GpuIvfRpq_FullDocNum_" + index_name);
    MONITOR_METRIC_WITH_INDEX(GpuIvfRpq_RtDocNum, "GpuIvfRpq_RtDocNum_" + index_name);
    index_name_ = index_name;
    char *pEnd;
    sort_mode_ = std::strtol(std::getenv("PARAM_SORT_MODE"), &pEnd, 10);

    LOG_INFO("End Init GpuIvfRpqSearcher, %s", index_name.c_str());

    return 0;
}

int GpuIvfRpqSearcher::LoadIndex(const std::string &path)
{
    // TODO
    return -1;
}

int GpuIvfRpqSearcher::LoadIndex(const void *data, size_t size)
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

    LOG_INFO("Created Neutron Index with rpq: %s", index_name.c_str());
    gpu_index_ = new neutron::gpu::NeutronIndexInterface(neutron::gpu::NeutronIndexInterface::CALCULATORTYPE::RPQ, 32,
                                                         32 * sizeof(uint8_t), device_no);
    neutron_manager_interface_ = GpuResourcesWrapper::GetInstance()->GetNeutronManagerInterface();
    gpu_index_->SetNeutronManagerInterface(neutron_manager_interface_);

    // 将索引中的向量全部拷贝到gpu device中
    gpu_index_->Add((const char *)index_->GetPqCodeProfile().getInfo(0), index_->GetDocNum());
    LOG_INFO("Doc num is %lu", index_->GetDocNum());
    // 将预计算表拷贝到gpu device中
    size_t lvl1_cent_num = index_->GetCentroidResourceManager().GetTotalCentroidsNum();
    LOG_INFO("Level1 centroids number is %lu", lvl1_cent_num);
    gpu_index_->AddTable((const char *)index_->GetCentRefineProfile().getInfo(0), lvl1_cent_num);
    own_gpu_index_ = true;
    return 0;
}

void GpuIvfRpqSearcher::SetIndex(Index::Pointer index)
{
    index_ = std::dynamic_pointer_cast<IvfRpqIndex>(index);
    dist_scorer_factory_.Init(index_->GetIndexMeta());
}

void GpuIvfRpqSearcher::SetBaseDocId(exdocid_t baseDocId)
{
    base_docid_ = baseDocId;
    index_->SetBaseDocid(baseDocId);
}

IndexMeta::FeatureTypes GpuIvfRpqSearcher::getFType()
{
    return this->index_->GetIndexMeta().type();
}

//! search by query
int GpuIvfRpqSearcher::Search(const QueryInfo &query_info, GeneralSearchContext *context)
{
    if (index_->MultiAgeMode() && query_info.MultiAgeMode()) {
        LOG_ERROR("GpuIvfRpq doesn't support multi age mode.");
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
    std::vector<CoarseIndex<SmallBlock>::PostingIterator> ivf_postings;
    std::vector<uint32_t> ivf_postings_group_ids;
    // (lvl1_index, lvl1_distance)
    std::vector<std::pair<uint32_t, distance_t>> lvl1_idx_dis;
    if (CollectIvfPostings(ivf_postings, ivf_postings_group_ids, group_doc_nums, 
                            query_info, real_slot_indexs, lvl1_idx_dis) != 0) {
        LOG_ERROR("collect ivf posting failed.");
        return -1;
    }

    if (lvl1_idx_dis.size() != ivf_postings.size()) {
        LOG_ERROR("lvl1_idx_dis size %lu mismatch with posting size %lu", lvl1_idx_dis.size(), ivf_postings.size());
        return -1;
    }

    // 2. 统计计算doc数量
    uint32_t total_node_count = 0;
    for (uint32_t i = 0; i < ivf_postings.size(); i++) {
        total_node_count += ivf_postings[i].getDocNum();
    }

    MONITOR_METRIC_LOG(GpuIvfRpq_GroupNum, group_num);
    if (base_docid_ == 0) {
        MONITOR_METRIC_LOG(GpuIvfRpq_FullDocNum, total_node_count);
    } else {
        MONITOR_METRIC_LOG(GpuIvfRpq_RtDocNum, total_node_count);
    }

    if (total_node_count == 0 || total_node_count >= 2000000) {
        LOG_WARN("dist_nodes size error: %u", total_node_count);
        timer.stop();
        transaction_GpuPreProcessTime(timer.u_elapsed(), false);
        return 0;
    }
    
    // 3. 取出倒排链下的点
    total_topk = std::min<size_t>(total_topk, total_node_count);
    std::vector<uint32_t> dist_nodes;
    std::vector<uint32_t> lvl1_indices;
    std::vector<distance_t> lvl1_dists;
    if (CollectIvfDistNodes(query_info, dist_nodes, lvl1_indices, lvl1_dists, lvl1_idx_dis,
                            ivf_postings, total_node_count, context) != 0) {
        LOG_ERROR("collect ivf dist nodes failed.");
        return -1;
    }

    // 检查
    for (uint32_t i = 0; i < dist_nodes.size(); i++) {
        if (dist_nodes[i] < 0 || dist_nodes[i] >= index_->GetDocNum()) {
            LOG_ERROR("dist nodes error: %u, max: %lu", dist_nodes[i], index_->GetDocNum());
            return -1;
        }
    }

    // 4. 初始化码本
    QueryDistanceMatrix1 qdm(index_->GetIndexMeta(), &(index_->GetPqCentroidResource()));
    if (!qdm.computeIpValMatrix(query_info.GetVector())) {
        LOG_ERROR("Init qdm IpVal matrix error");
        return -1;
    }

    // 检查
    float *codebook = (float *)qdm.GetIpValArray();
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
    if (gpu_index_->SearchTable(distances, labels,
                                dist_nodes, lvl1_indices,
                                lvl1_dists, total_topk,
                                qdm.GetIpValArray(), 32 * 256) != 0) 
    {
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
            LOG_ERROR("final result failed, %s, %d, %lu, %d, %s", index_name_.c_str(), sort_mode_, i, labels[i],
                        const_cast<QueryInfo &>(query_info).GetRawQuery().c_str());
            return -1;
        }
        context->emplace_back(0, sort_mode_ ? labels[i] : dist_nodes[labels[i]], distances[i]);
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

int GpuIvfRpqSearcher::CollectIvfPostings(std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings,
                                              std::vector<uint32_t> &ivf_postings_group_ids,
                                              std::vector<uint32_t> &group_doc_nums, const QueryInfo &query_info,
                                              std::vector<std::vector<off_t>> &real_slot_indexs,
                                              std::vector<std::pair<uint32_t, distance_t>>& lvl1_idx_dis)
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

            if (index_->SearchIvf(group_index,
                                  query_info.MultiQueryMode() ? query_info.GetVectors().at(i)
                                                              : query_info.GetVectors().at(0),
                                  query_info.GetVectorLen(), query_info.GetDimension(), query_info.GetContextParams(),
                                  real_slot_indexs[i], lvl1_idx_dis) != 0) {
                LOG_ERROR("Failed to call SearchIvf.");
                return -1;
            }
        }
    }
    RecoverPostingFromSlot(ivf_postings, ivf_postings_group_ids, real_slot_indexs, group_doc_nums, false,
                           query_info.MultiQueryMode());
    return 0;
}

void GpuIvfRpqSearcher::RecoverPostingFromSlot(std::vector<CoarseIndex<SmallBlock>::PostingIterator> &postings,
                                                std::vector<uint32_t> &ivf_postings_group_ids,
                                                std::vector<std::vector<off_t>> &real_slot_indexs,
                                                std::vector<uint32_t> &group_doc_nums, bool need_truncate,
                                                bool is_multi_query)
{
    const CoarseIndex<SmallBlock> &coarse_index = index_->GetCoarseIndex();
    for (size_t i = 0; i < real_slot_indexs.size(); ++i) {
        postings.reserve(postings.size() + real_slot_indexs[i].size());
        uint32_t posting_num = 0;
        for (auto &group_real_slot_index : real_slot_indexs[i]) {
            CoarseIndex<SmallBlock>::PostingIterator posting = coarse_index.search(group_real_slot_index);
            posting_num += posting.getDocNum();
            group_doc_nums[i] += posting.getDocNum();
            postings.push_back(posting);
            ivf_postings_group_ids.push_back(is_multi_query ? i : 0);
        }
    }
}

int GpuIvfRpqSearcher::CollectIvfDistNodes(const QueryInfo &query_info, std::vector<uint32_t> &dist_nodes,
                                            std::vector<uint32_t> &lvl1_indices, std::vector<distance_t> &lvl1_dists,
                                            std::vector<std::pair<uint32_t, distance_t>>& lvl1_idx_dis,
                                            std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings,
                                            uint32_t total_node_count, GeneralSearchContext *context)
{
    dist_nodes.resize(total_node_count);
    lvl1_indices.resize(total_node_count);
    lvl1_dists.resize(total_node_count);
    const std::vector<GroupInfo> &group_infos = query_info.GetGroupInfos();
    const size_t group_nums = group_infos.size();
    auto &real_slot_indexs = context->getAllGroupRealSlotIndexs();
    
    size_t node_index = 0;
    for (size_t i = 0; i < ivf_postings.size(); i++) {
        CoarseIndex<SmallBlock>::PostingIterator &iter = ivf_postings.at(i);
        while (UNLIKELY(!iter.finish())) {
            uint32_t docid = iter.next();
            dist_nodes.at(node_index) = docid;
            lvl1_indices.at(node_index) = lvl1_idx_dis[i].first;
            lvl1_dists.at(node_index) = lvl1_idx_dis[i].second;
            node_index++;
        }
    }
    
    return 0;
}

int GpuIvfRpqSearcher::PostProcess(const QueryInfo &query_info, GeneralSearchContext *context,
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
            if (index_->EnableQuantize()) {
                float vmin = index_->GetVmin();
                float vmax = index_->GetVmax();
                size_t d = index_->GetIndexMeta().dimension();

                std::vector<float> decoded_vector(d);
                decode_vector(static_cast<const uint8_t*>(feature), decoded_vector.data(), vmin, vmax, d);
                
                dist = scorer.Score(decoded_vector.data(), query_info.GetVector());
            } else {
                if (likely(query_info.GetVectors().size() == 1)) {
                    dist = scorer.Score(feature, query_info.GetVector());
                } else {
                    dist = scorer.Score(feature, query_info.GetVectors().at(result.poolId));
                }
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