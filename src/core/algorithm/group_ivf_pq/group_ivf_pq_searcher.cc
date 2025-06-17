#include "group_ivf_pq_searcher.h"
#include "src/core/framework/utility/closure.h"
#include "src/core/algorithm/thread_common.h"
#include "bthread/bthread.h"

MERCURY_NAMESPACE_BEGIN(core);

void* GroupIvfPqSearcher::BthreadRun(void* message) {
    SearcherMessage* msg = static_cast<SearcherMessage*>(message);
    if (msg->searcher && msg->qdms) {
        msg->searcher->BatchScorePq(msg->node_start, msg->node_end,
                                    msg->dist_nodes, *msg->query_info,
                                    msg->qdms);
    } else if (msg->searcher) {
        msg->searcher->BatchScore(msg->node_start, msg->node_end,
                                  msg->dist_nodes, *msg->query_info);
    }
    return nullptr;
}

void GroupIvfPqSearcher::BatchScore(size_t node_start, size_t node_end,
                                    std::vector<DistNode>* dist_nodes,
                                    const QueryInfo& query_info) const {
    //召回率计算模式，普通距离计算
    OrigDistScorer scorer = dist_scorer_factory_.Create();
    for (size_t i = node_start; i < node_end; i++) {
        DistNode& dist_node = dist_nodes->at(i);
        docid_t docid = dist_node.key;
        const void * feature = nullptr;
        if (unlikely(vector_retriever_.isValid())) {
            if (!vector_retriever_(base_docid_ + docid, feature)) {
                LOG_ERROR("retrieve vector failed. docid:%u", docid);
                continue;
            }
        } else {
            feature = index_->GetFeatureProfile().getInfo(docid);
        }

        if (feature == nullptr) {
            LOG_ERROR("get null feature. docid: %u", docid);
            continue;
        }

        float dist = 0.0;
        if (likely(query_info.GetVectors().size() == 1)) {
            dist = scorer.Score(feature, query_info.GetVector());
        } else {
            dist = scorer.Score(feature, query_info.GetVectors());
        }

        dist_node.dist = dist;
    }
}

void GroupIvfPqSearcher::BatchScorePq(size_t node_start, size_t node_end,
                                      std::vector<DistNode>* dist_nodes,
                                      const QueryInfo& query_info,
                                      std::vector<QueryDistanceMatrix *>* qdms) const {
    
    PqDistScorer scorer(&index_->GetPqCodeProfile());
    for (size_t i = node_start; i < node_end; i++) {
        DistNode& dist_node = dist_nodes->at(i);
        dist_node.dist = scorer.score(dist_node.key, qdms->at(query_info.MultiQueryMode() ? dist_node.offset : 0));
    }
}

void GroupIvfPqSearcher::BatchScoreRPq(size_t node_start, size_t node_end,
                                      std::vector<DistNode>* dist_nodes,
                                      const QueryInfo& query_info,
                                      QueryDistanceMatrix* qdm) const {
    
    PqDistScorer scorer(&index_->GetPqCodeProfile());
    for (size_t i = node_start; i < node_end; i++) {
        DistNode& dist_node = dist_nodes->at(i);
        dist_node.dist = scorer.score(dist_node.key, qdm);
    }
}


size_t GroupIvfPqSearcher::GetPostingsNodeCount(const std::vector<CoarseIndex<BigBlock>::PostingIterator>& ivf_postings,
                                              size_t start, size_t end) const {
    size_t count = 0;
    for (size_t i = start; i < end; i++) {
        count += ivf_postings.at(i).getDocNum();
    }

    return count;
}

GroupIvfPqSearcher::GroupIvfPqSearcher() {
    if (!index_) {
        index_.reset(new GroupIvfPqIndex());
    }
    SetThreadEnv();
}

int GroupIvfPqSearcher::Init(IndexParams &params) {
    index_->SetIndexParams(params);

    std::string index_name = params.getString(PARAM_VECTOR_INDEX_NAME);
    LOG_INFO("Start Init GroupIvfPqSearcher, %s", index_name.c_str());

    MONITOR_TRANSACTION(GroupIvfPq, CollectIvfPostings);
    MONITOR_TRANSACTION(GroupIvfPq, CalcNodeDist);
    MONITOR_TRANSACTION(GroupIvfPq, CollectGroupHeaps);
    MONITOR_TRANSACTION(GroupIvfPq, CollectBasicResult);
    MONITOR_TRANSACTION(GroupIvfPq, CollectLeftResult);
    MONITOR_TRANSACTION(GroupIvfPq, CollectNthTopk);
    MONITOR_TRANSACTION(GroupIvfPq, PostProcess);
    MONITOR_TRANSACTION(GroupIvfPq, CollectMultiAgeResult);
    MONITOR_TRANSACTION(GroupIvfPq, StatAgeInfo);
    MONITOR_TRANSACTION(GroupIvfPq, GenerateAgeSortedContainer);
    MONITOR_TRANSACTION(GroupIvfPq, SortInEachAge);
    MONITOR_METRIC_WITH_INDEX(GroupIvfPq_CentroidNum, "GroupIvfPq_CentroidNum_" + index_name);
    MONITOR_METRIC_WITH_INDEX(GroupIvfPq_GroupNum, "GroupIvfPq_GroupNum_" + index_name);
    MONITOR_METRIC_WITH_INDEX(GroupIvfPq_FullDocNum, "GroupIvfPq_FullDocNum_" + index_name);
    MONITOR_METRIC_WITH_INDEX(GroupIvfPq_RtDocNum, "GroupIvfPq_RtDocNum_" + index_name);

    return 0;
}

int GroupIvfPqSearcher::LoadIndex(const std::string& path) {
    //TODO
    return -1;
}

int GroupIvfPqSearcher::LoadIndex(const void* data, size_t size) {
    if (index_->Load(data, size) != 0) {
        LOG_ERROR("Failed to load index.");
        return -1;
    }
    dist_scorer_factory_.Init(index_->GetIndexMeta());
    return 0;
}

void GroupIvfPqSearcher::SetIndex(Index::Pointer index)
{
    index_ = std::dynamic_pointer_cast<GroupIvfPqIndex>(index);
    dist_scorer_factory_.Init(index_->GetIndexMeta());
}

void GroupIvfPqSearcher::SetBaseDocId(exdocid_t baseDocId)
{
    base_docid_ = baseDocId;
    index_->SetBaseDocid(baseDocId);
}

IndexMeta::FeatureTypes GroupIvfPqSearcher::getFType()
{
    return this->index_->GetIndexMeta().type();
}

int GroupIvfPqSearcher::CollectMultiAgeResult(const QueryInfo& query_info,
                                              std::vector<DistNode>& dist_nodes,
                                              GeneralSearchContext* context) const {
    /*
        1. 遍历所有doc统计所属的笔记年龄段信息，以及汇总每个笔记年龄段包含的笔记数量
        2. 所有笔记共享一个container，但container内需要按年龄段分段存放，因此初始化各年龄段cursors指向各自应该在的起始地址
        3. 复用原有container，再次遍历所有doc，并不断交换不属于当前位置年龄段的笔记到应该在的年龄段
        4. 每个年龄分段内通过nth_element挑选topk，并推入最终结果
    */
    /*
        TODO:
        1. 统计信息提前到到CalcNodeDist内，超出最大时间范围和被删除的笔记提前丢弃不参与计算
        2. 当前超出最大时间范围的笔记直接丢弃，极小概率会导致召回不足，是否尝试单独存储一个队列，召回不足时从该队列补充
    */
    //定时器
    butil::Timer timer;
    timer.start();
    const std::vector<AgeInfo>& age_infos = query_info.GetAgeInfos();
    // 1. stat age info
    std::vector<uint32_t> age_counters(age_infos.size(), 0);
    int64_t now_timestamp_s = butil::gettimeofday_s();
    for (size_t i = 0; i < dist_nodes.size(); ++i) {
        uint32_t create_timestamp_s = *(uint32_t*)index_->GetDocCreateTimeProfile().getInfo(dist_nodes[i].key);
        uint32_t doc_age = now_timestamp_s - create_timestamp_s;
        dist_nodes[i].offset = -1;
        for (size_t j = 0; j < age_infos.size(); ++j) {
            if (doc_age <= age_infos[j].age) {
                age_counters[j]++;
                dist_nodes[i].offset = j;
                break;
            }
        }
    }
    timer.stop();
    transaction_StatAgeInfo(timer.u_elapsed(), true);
    // 2. init each age cursor, last element means last age end
    std::vector<uint32_t> age_cursors(age_infos.size() + 1, 0);
    for (size_t i = 1; i < age_cursors.size(); ++i ) {
        age_cursors[i] = age_cursors[i - 1] + age_counters[i - 1];
    }
    timer.start();
    // 3. generate age sorted dist node container
    std::vector<uint32_t> ages_end(age_cursors.begin() + 1, age_cursors.end());
    for (size_t i = 0; i < age_infos.size(); ++i) {
        size_t left = age_cursors[i], right = ages_end[i];
        while (left < right) {
            if (dist_nodes[left].offset == -1) {
                std::swap(dist_nodes[left], dist_nodes[age_cursors[age_infos.size()]++]);
            } else if (dist_nodes[left].offset == static_cast<int>(i)) {
                ++left;
                ++age_cursors[i];
            } else {
                uint32_t doc_age = dist_nodes[left].offset;
                std::swap(dist_nodes[left], dist_nodes[age_cursors[doc_age]]);
                ++age_cursors[doc_age];
            }
        }
    }
    timer.stop();
    transaction_GenerateAgeSortedContainer(timer.u_elapsed(), true);
    timer.start();
    // 4. sort in each age range
    uint32_t start, end = 0, topk;
    for (size_t i = 0; i < age_infos.size(); ++i) {
        start = end;
        end = age_cursors[i];
        topk = age_infos[i].topk;
        CollectNthTopk(start, end, topk, dist_nodes, context, i);
    }
    timer.stop();
    transaction_SortInEachAge(timer.u_elapsed(), true);
    return 0;
}

int GroupIvfPqSearcher::CollectNthTopk(uint32_t start, uint32_t end, uint32_t topk,
                                       std::vector<DistNode>& dist_nodes,
                                       GeneralSearchContext* context,
                                       uint32_t offset) const {
    if (end > start + topk) {
        std::nth_element(dist_nodes.begin() + start,
                         dist_nodes.begin() + start + topk,
                         dist_nodes.begin() + end);
    }

    context->Result().reserve(context->Result().size() + topk);
    bool with_pk = index_->WithPk();
    for (size_t i = start; i < start + topk && i < end; i++) {
        const DistNode& node = dist_nodes[i];
        pk_t pk = 0;
        if (unlikely(with_pk)) {
            pk = index_->GetPk(node.key);
        }
        docid_t glo_doc_id = node.key + base_docid_;
        // TODO: move to before sort
        if (deletion_map_retriever_.isValid() && deletion_map_retriever_(glo_doc_id)) {
            continue;
        }
        context->emplace_back(pk, glo_doc_id, node.dist, offset);
    }
    
    return 0;
}

int GroupIvfPqSearcher::Search(const QueryInfo& query_info, GeneralSearchContext* context) {

    // if (index_->EnableResidual()) {
    //     std::cout << "GroupIvfPqSearcher::Search EnableResidual" << std::endl;
    // }

    const std::vector<GroupInfo>& group_infos = query_info.GetGroupInfos();

    if (query_info.MultiQueryMode()) {
        if (query_info.GetVectors().size() != group_infos.size()) {
            LOG_ERROR("num of query vector is not equal to group, %lu != %lu",
                      query_info.GetVectors().size(),
                      group_infos.size());
            return false;
        }
    }

    if (query_info.GetDimension() != index_->GetIndexMeta().dimension()) {
        LOG_ERROR("query dimension %lu != index dimension %lu.", query_info.GetDimension(), index_->GetIndexMeta().dimension());
        return -1;
    }

    //定时器
    butil::Timer timer;

    std::vector<std::vector<off_t>>& real_slot_indexs = context->getAllGroupRealSlotIndexs();
    std::vector<uint32_t> group_doc_nums(group_infos.size(), 0); // 每个group有多少个doc
    std::vector<CoarseIndex<BigBlock>::PostingIterator> ivf_postings;
    std::vector<uint32_t> ivf_postings_group_ids;
    // (lvl1_index, lvl2_index)
    std::vector<std::pair<off_t, off_t>> level_indexs;
    MONITOR_TRANSACTION_LOG(CollectIvfPostings(ivf_postings, ivf_postings_group_ids, 
                            group_doc_nums, query_info, real_slot_indexs, level_indexs), 
            "collect ivf posting failed.",
            GroupIvfPq,
            CollectIvfPostings)

    MONITOR_METRIC_LOG(GroupIvfPq_CentroidNum, ivf_postings.size());
    MONITOR_METRIC_LOG(GroupIvfPq_GroupNum, group_infos.size());

    std::vector<DistNode> dist_nodes;
    MONITOR_TRANSACTION_LOG(CalcNodeDist(ivf_postings, ivf_postings_group_ids, query_info, dist_nodes, level_indexs), 
            "calc node dist failed.",
            GroupIvfPq,
            CalcNodeDist)

    if (index_->MultiAgeMode() && query_info.MultiAgeMode()) {
        // TODO: implement multi age sort
        MONITOR_TRANSACTION_LOG(CollectMultiAgeResult(query_info, dist_nodes, context),
                "collect multi age result failed.",
                GroupIvfPq,
                CollectMultiAgeResult);
        MONITOR_TRANSACTION_LOG(PostProcess(query_info, context, 1),
                "post process result failed.",
                GroupIvfPq,
                PostProcess);
        return 0;
    } else if (!index_->MultiAgeMode() && query_info.MultiAgeMode()) {
        LOG_WARN("index is not multi age mode, but query is, will recall from 0:0 group");
    }

    if (query_info.GetGroupInfos().size() == 1) {
        uint32_t total = query_info.GetTotalRecall();
        if (total == 0 && !query_info.GetTopks().empty()) {
            total = query_info.GetTopks()[0];
        }
        MONITOR_TRANSACTION_LOG(CollectNthTopk(0, dist_nodes.size(), total, dist_nodes, context, 0),
                "collect nth topk failed.",
                GroupIvfPq,
                CollectNthTopk);

        // std::cout << "dist_nodes[0].key = " << dist_nodes[0].key << "; dist_nodes[0].dist = " << dist_nodes[0].dist << std::endl;
        // std::cout << "dist_nodes[1].key = " << dist_nodes[1].key << "; dist_nodes[1].dist = " << dist_nodes[1].dist << std::endl;
        // std::cout << "dist_nodes[2].key = " << dist_nodes[2].key << "; dist_nodes[2].dist = " << dist_nodes[2].dist << std::endl;
        MONITOR_TRANSACTION_LOG(PostProcess(query_info, context, group_infos.size()),
                "post process result failed.",
                GroupIvfPq,
                PostProcess);
        return 0;
    }

    MONITOR_TRANSACTION_LOG(CollectGroupHeaps(query_info, dist_nodes, group_doc_nums, context), 
                "collect group heaps failed.",
                GroupIvfPq,
                CollectGroupHeaps)

    MONITOR_TRANSACTION_LOG(PostProcess(query_info, context, group_infos.size()),
                "post process result failed.",
                GroupIvfPq,
                PostProcess);

    return 0;
}

int GroupIvfPqSearcher::CalcNodeDist(std::vector<CoarseIndex<BigBlock>::PostingIterator>& ivf_postings,
                                   std::vector<uint32_t>& ivf_postings_group_ids,
                                   const QueryInfo& query_info,
                                   std::vector<DistNode>& dist_nodes,
                                   std::vector<std::pair<off_t, off_t>>& level_indexs) {
    if (ivf_postings.size() == 0) {
        LOG_WARN("ivf_postings size is zero");
        return 0;
    }

    size_t node_count = GetPostingsNodeCount(ivf_postings, 0, ivf_postings.size());
    if (base_docid_ == 0) {
        MONITOR_METRIC_LOG(GroupIvfPq_FullDocNum, node_count);
    } else {
        MONITOR_METRIC_LOG(GroupIvfPq_RtDocNum, node_count);
    }
    //dist_nodes.resize(node_count);
    dist_nodes.reserve(node_count);
    
    bool is_use_pq = true;
    if (query_info.GetContextParams().has(PARAM_GENERAL_RECALL_TEST_MODE) && query_info.GetContextParams().has(PARAM_COARSE_SCAN_RATIO)) { //测召回模式，全走normal距离
        auto ratio = query_info.GetContextParams().getFloat(PARAM_COARSE_SCAN_RATIO);
        if (std::abs(ratio - 1.0) < 1e-6) {
            is_use_pq = false;
        }
    }

    if (index_->EnableResidual() && is_use_pq) {
        if (level_indexs.size() != ivf_postings.size()) {
            LOG_ERROR("level size %lu mismatch with posting size %lu", level_indexs.size(), ivf_postings.size());
            return -1;
        }
        const std::vector<const void *>& query_infos = query_info.GetVectors();
        if (query_infos.size() != 1) {
            LOG_ERROR("multi-query not supported for RPQ!");
            return -1;
        }
        size_t node_start_ = 0;
        size_t node_end_ = 0;
        const void *centroid_value = nullptr;
        for (size_t i = 0; i < ivf_postings.size(); i++) {
            // std::cout << ivf_postings[i].getDocNum() << " ; (" << level_indexs[i].first << "," << level_indexs[i].second << ")" << std::endl;
            if (ivf_postings[i].getDocNum() != 0) {
                node_end_ += ivf_postings[i].getDocNum();
                // std::cout << "node_start_ = " << node_start_ << "; node_end_ = " << node_end_ << std::endl;
                CoarseIndex<BigBlock>::PostingIterator& iter = ivf_postings.at(i);
                while (UNLIKELY(!iter.finish())) {
                    docid_t docid = iter.next();
                    dist_nodes.emplace_back(docid, 0, ivf_postings_group_ids.at(i));
                }
                QueryDistanceMatrix* qdm = new QueryDistanceMatrix(index_->GetIndexMeta(), &(index_->GetPqCentroidResource()));
                if (!index_->EnableFineCluster()) {
                    // std::cout << "lvl1" << std::endl;
                    centroid_value = 
                    index_->GetCentroidResourceManager().GetCentroidResource(0).getValueInRoughMatrix(0, level_indexs[i].first);
                } else {
                    // std::cout << "lvl2" << std::endl;
                    const mercury::core::CentroidResource &fine_centroid_resource = 
                    index_->GetFineCentroidResourceManager().GetCentroidResource(level_indexs[i].first);
                    centroid_value = fine_centroid_resource.getValueInRoughMatrix(0, level_indexs[i].second);
                }
                if (index_->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
                    // std::cout << "half" << std::endl;
                    std::vector<half_float::half> residual(index_->GetIndexMeta().dimension());
                    for (size_t j = 0; j < index_->GetIndexMeta().dimension(); j++) {
                        residual[j] = (half_float::half)*((half_float::half *)(query_info.GetVector()) + j) 
                                    - (half_float::half)*((half_float::half *)centroid_value + j);
                        // std::cout << residual[j] << " ";
                    }
                    // std::cout << std::endl;
                    if (!qdm->initDistanceMatrix(residual.data())) {
                        LOG_ERROR("Init qdm distance matrix error");
                        return -1;
                    }
                } else if (index_->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeFloat) {
                    // std::cout << "float" << std::endl;
                    std::vector<float> residual(index_->GetIndexMeta().dimension());
                    for (size_t j = 0; j < index_->GetIndexMeta().dimension(); j++) {
                        residual[j] = (float)*((float *)(query_info.GetVector()) + j) 
                                    - (float)*((float *)centroid_value + j);
                        // std::cout << "residual[" << j << "] = " << (float)*((float *)(query_info.GetVector()) + j) 
                        // << " - " << (float)*((float *)centroid_value + j) << "; ";
                    }
                    // std::cout << std::endl;
                    if (!qdm->initDistanceMatrix(residual.data())) {
                        LOG_ERROR("Init qdm distance matrix error");
                        return -1;
                    }
                } else {
                    LOG_ERROR("Unsupported Type!");
                    return -1;
                }
                BatchScoreRPq(node_start_, node_end_, &dist_nodes, query_info, qdm);
                delete qdm;
                node_start_ = node_end_;
            }
        }
        return 0;
    }

    //std::vector<size_t> ivf_postings_docids;
    for (size_t i = 0; i < ivf_postings.size(); i++) {
        CoarseIndex<BigBlock>::PostingIterator& iter = ivf_postings.at(i);
        while (UNLIKELY(!iter.finish())) {
            docid_t docid = iter.next();
            dist_nodes.emplace_back(docid, 0, ivf_postings_group_ids.at(i));
        }
    }

    PartitionStrategy strategy(node_count, mercury_doc_num_per_concurrency, mercury_max_concurrency_num, mercury_need_parallel);
    PartitionStrategy::return_type partition_result;
    if (strategy.MixedPartitionByDoc(node_count, partition_result) != 0) {
        LOG_ERROR("concurrency partition failed");
        return -1;
    }

    const std::vector<const void *>& query_infos = query_info.GetVectors();

    //初始化PQ距离向量
    std::vector<QueryDistanceMatrix *> qdms;
    qdms.reserve(query_infos.size());
    for (size_t i = 0; i < query_infos.size(); i++) {
        QueryDistanceMatrix* qdm = new QueryDistanceMatrix(index_->GetIndexMeta(), &(index_->GetPqCentroidResource()));
        if (is_use_pq && !qdm->initDistanceMatrix(query_info.GetVectors().at(i))) {
            LOG_ERROR("Init qdm distance matrix error");
            return -1;
        }
        qdms.push_back(qdm);
    }

    std::vector<bthread_t> bthreads;
    std::vector<SearcherMessage> msgs;
    msgs.resize(partition_result.size());

    for (size_t i = 0; i < partition_result.size() - 1; i++) {
        SearcherMessage& message = msgs.at(i);
        bthread_t bid;
        message.searcher = this;
        message.dist_nodes = &dist_nodes;
        message.node_start = partition_result.at(i).node_start_;
        message.node_end = partition_result.at(i).posting_end_;
        message.query_info = &query_info;
        if (is_use_pq) {
            message.qdms = &qdms;
        }

        if (bthread_start_background(&bid, NULL, BthreadRun, &msgs.at(i)) != 0) {
            LOG_ERROR("start bthread failed.");
            return -1;
        }

        bthreads.push_back(bid);
    }

    Partition& last_partition = partition_result.at(partition_result.size() - 1);
    if (is_use_pq) {
        BatchScorePq(last_partition.node_start_, last_partition.posting_end_, &dist_nodes, query_info, &qdms);
    } else {
        BatchScore(last_partition.node_start_, last_partition.posting_end_, &dist_nodes, query_info);
    }

    for (auto t : bthreads) {
        bthread_join(t, NULL);
    }

    for (auto qdm : qdms) {
        if (qdm) {
            delete qdm;
        }
    }
    qdms.clear();

    return 0;
}

int GroupIvfPqSearcher::CollectIvfPostings(std::vector<CoarseIndex<BigBlock>::PostingIterator>& ivf_postings,
                                          std::vector<uint32_t>& ivf_postings_group_ids,
                                          std::vector<uint32_t>& group_doc_nums,
                                          const QueryInfo& query_info,
                                          std::vector<std::vector<off_t>>& real_slot_indexs,
                                          std::vector<std::pair<off_t, off_t>>& level_indexs) {
    const std::vector<GroupInfo>& group_infos = query_info.GetGroupInfos();

    if (real_slot_indexs.empty()) {
        real_slot_indexs.resize(group_infos.size());
        for (size_t i = 0; i < group_infos.size(); i++) {
            gindex_t group_index = index_->GetGroupManager().GetGroupIndex(group_infos.at(i));
            if (group_index == INVALID_GROUP_INDEX) {
                LOG_WARN("group not in group manager. level:%d, id:%d",
                    group_infos.at(i).level, group_infos.at(i).id);
                continue;
            }

            if (index_->SearchIvf(group_index, query_info.MultiQueryMode() ? query_info.GetVectors().at(i) : query_info.GetVectors().at(0),
                                  query_info.GetVectorLen(), query_info.GetDimension(),
                                  query_info.GetContextParams(), real_slot_indexs[i], level_indexs) != 0) {
                LOG_ERROR("Failed to call SearchIvf.");
                return -1;
            }

        }
    }
    index_->RecoverPostingFromSlot(ivf_postings, ivf_postings_group_ids, real_slot_indexs, group_doc_nums, query_info.MultiQueryMode());
    return 0;
}

// TODO: ensure group quota
int GroupIvfPqSearcher::CollectGroupHeaps(const QueryInfo& query_info,
                                          std::vector<DistNode>& dist_nodes,
                                          const std::vector<uint32_t>& group_doc_nums,
                                          GeneralSearchContext* context) const {
    uint32_t total = query_info.GetTotalRecall();
    if (total == 0) {
        for (size_t i = 0; i < query_info.GetTopks().size(); i++) {
            total += query_info.GetTopks()[i];
        }
    }
    context->Result().reserve(total);
    uint32_t group_start = 0;
    for (size_t i = 0; i < query_info.GetGroupInfos().size(); i++) {
        uint32_t topk = query_info.GetTopks().at(i);
        uint32_t group_doc_num = group_doc_nums.at(i);
        if (topk < group_doc_num) {
            std::nth_element(dist_nodes.begin() + group_start,
                             dist_nodes.begin() + group_start + topk,
                             dist_nodes.begin() + group_start + group_doc_num);
        }

        for (size_t j = group_start; j < group_start + topk && j < group_start + group_doc_num; j++) {
            auto& node = dist_nodes[j];
            docid_t glo_doc_id = node.key + base_docid_;
            context->emplace_back(0, glo_doc_id, node.dist, i);
        }
        group_start += group_doc_num;
    }
    return 0;
}

int GroupIvfPqSearcher::CollectBasicResult(std::vector<MyHeap<DistNode>>& group_heaps, const std::vector<uint32_t>& topks,
                                        GeneralSearchContext* context) const {
    // 先从每个堆pop topk个，组成结果集1
    bool with_pk = index_->WithPk();
    size_t capacity = 0;
    for (size_t i = 0; i < group_heaps.size(); i++) {
        capacity += topks.at(i);
    }
    context->Result().reserve(capacity);

    for (size_t i = 0; i < group_heaps.size(); i++) {
        MyHeap<DistNode>& result = group_heaps.at(i);
        uint32_t topk = topks.at(i);
        for (size_t j = 0; j < topk && !result.fetchEnd(); j++) {
            const DistNode* node = result.fetch();
            if (node == nullptr) {
                LOG_ERROR("fetch node null.");
                return -1;
            }

            pk_t pk = 0;
            if (unlikely(with_pk)) {
                pk = index_->GetPk(node->key);
            }

            docid_t glo_doc_id = node->key + base_docid_;
            if (deletion_map_retriever_.isValid() && deletion_map_retriever_(glo_doc_id)) {
                continue;
            }
            context->emplace_back(pk, glo_doc_id, node->dist);
        }
    }

    return 0;
}

int GroupIvfPqSearcher::CollectLeftResult(uint32_t total, std::vector<MyHeap<DistNode>>& group_heaps,
                                        GeneralSearchContext* context) const {
    int64_t left_count = total - context->Result().size();
    if (total > 0 && left_count > 0) {
        MyHeap<DistNode> left_result(left_count);
        for (size_t i = 0; i < group_heaps.size(); i++) {
            MyHeap<DistNode>& group_heap = group_heaps.at(i);
            PushToHeap(group_heap, left_result);
        }

        PushSearchResultToContext(context, left_result.getData());
    }

    return 0;
}

int GroupIvfPqSearcher::PostProcess(const QueryInfo& query_info, GeneralSearchContext* context, 
                                    size_t group_num) const {
    
    std::vector<SearchResult> &results = context->Result();

    // calculate real similarity score if doc features are saved
    if (index_->ContainFeature() || vector_retriever_.isValid()) {
        OrigDistScorer scorer = dist_scorer_factory_.Create();
        for (auto &result : results) {
            const void * feature = nullptr;
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


void GroupIvfPqSearcher::PushSearchResultToContext(GeneralSearchContext* context, const std::vector<DistNode>& dist_vec) const {
    bool with_pk = index_->WithPk();
    context->Result().reserve(context->Result().size() + dist_vec.size());
    for (size_t i = 0; i < dist_vec.size(); i++) {
        const DistNode& node = dist_vec.at(i);
        pk_t pk = 0;
        if (unlikely(with_pk)) {
            pk = index_->GetPk(node.key);
        }

        docid_t glo_doc_id = node.key + base_docid_;
        if (deletion_map_retriever_.isValid() && deletion_map_retriever_(glo_doc_id)) {
            continue;
        }
        context->emplace_back(pk, glo_doc_id, node.dist);
    }
}

MERCURY_NAMESPACE_END(core);
