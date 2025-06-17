#include "group_ivf_searcher.h"
#include "src/core/framework/utility/closure.h"
#include "src/core/algorithm/thread_common.h"
#include "src/core/algorithm/partition_strategy.h"
#include "bthread/bthread.h"
#include "putil/src/putil/StringUtil.h"

MERCURY_NAMESPACE_BEGIN(core);

void* GroupIvfSearcher::BthreadRun(void* message) {
    SearcherMessage* msg = static_cast<SearcherMessage*>(message);
    if (msg->searcher) {
        msg->searcher->BatchScore(msg->node_start, msg->node_end, 
                                  *msg->dist_nodes, *msg->query_info);
    }
    return nullptr;
}

void GroupIvfSearcher::BatchScore(size_t node_start, size_t node_end,
                                  std::vector<DistNode>& dist_nodes,
                                  const QueryInfo& query_info) const {
    OrigDistScorer scorer = dist_scorer_factory_.Create();
    for (size_t i = node_start; i < node_end; i++) {
        const void* query = query_info.GetVectors().at(dist_nodes[i].offset);
        docid_t docid = dist_nodes[i].key;
        const void* goods = nullptr;
        if (unlikely(vector_retriever_.isValid() && !index_->ContainFeature())) {
            if (!vector_retriever_(base_docid_ + docid, goods)) {
                LOG_ERROR("retrieve vector failed. docid:%u", docid);
                continue;
            }
        } else {
            goods = index_->GetFeatureProfile().getInfo(docid);
        }

        if (goods == nullptr) {
            LOG_ERROR("get null goods. docid: %u", docid);
            continue;
        }

        float dist = 0.0;
        dist = scorer.Score(goods, query);

        dist_nodes[i].dist = dist;
    }

    //promise.set_value(0);
}

size_t GroupIvfSearcher::GetPostingsNodeCount(const std::vector<CoarseIndex<SmallBlock>::PostingIterator>& ivf_postings,
                                              size_t start, size_t end) const {
    size_t count = 0;
    for (size_t i = start; i < end; i++) {
        count += ivf_postings.at(i).getDocNum();
    }

    return count;
}

GroupIvfSearcher::GroupIvfSearcher() {
    if (!index_) {
        index_.reset(new GroupIvfIndex());
    }
    SetThreadEnv();
}

int GroupIvfSearcher::Init(IndexParams &params) {
    index_->SetIndexParams(params);
        
    index_name_ = params.getString(PARAM_VECTOR_INDEX_NAME);
    delmap_before = params.getBool(PARAM_DELMAP_BEFORE);
    LOG_INFO("Start Init GroupIvfSearcher, %s", index_name_.c_str());

    MONITOR_TRANSACTION(GroupIvf, CollectIvfPostings);
    MONITOR_TRANSACTION(GroupIvf, CalcNodeDist);
    MONITOR_TRANSACTION(GroupIvf, CollectMultiAgeResult);
    MONITOR_TRANSACTION(GroupIvf, StatAgeInfo);
    MONITOR_TRANSACTION(GroupIvf, GenerateAgeSortedContainer);
    MONITOR_TRANSACTION(GroupIvf, SortInEachAge);
    MONITOR_TRANSACTION(GroupIvf, CollectNthTopk);
    MONITOR_TRANSACTION(GroupIvf, CollectGroupHeaps);
    MONITOR_TRANSACTION(GroupIvf, CollectBasicResult);
    MONITOR_TRANSACTION(GroupIvf, CollectLeftResult);
    MONITOR_TRANSACTION(GroupIvf, PostProcess);
    MONITOR_METRIC(GroupIvf_CentroidNum);
    MONITOR_METRIC(GroupIvf_GroupNum);
    MONITOR_METRIC_WITH_INDEX(GroupIvf_FullDocNum, "GroupIvf_FullDocNum_" + index_name_);
    MONITOR_METRIC_WITH_INDEX(GroupIvf_RtDocNum, "GroupIvf_RtDocNum_" + index_name_);

    LOG_INFO("End Init GroupIvfSearcher, %s", index_name_.c_str());

    return 0;
}

int GroupIvfSearcher::LoadIndex(const std::string& path) {
    //TODO
    return -1;
}

int GroupIvfSearcher::LoadIndex(const void* data, size_t size) {
    if (index_->Load(data, size) != 0) {
        LOG_ERROR("Failed to load index.");
        return -1;
    }
    if (index_->ForceHalf()) {
        auto half_index_meta = index_->GetIndexMeta();
        half_index_meta.setType(IndexMeta::kTypeHalfFloat);
        if (half_index_meta.method() == mercury::core::IndexDistance::kMethodFloatSquaredEuclidean) {
            half_index_meta.setMethod(mercury::core::IndexDistance::kMethodHalfFloatSquaredEuclidean);
        } else if (half_index_meta.method() == mercury::core::IndexDistance::kMethodFloatInnerProduct) {
            half_index_meta.setMethod(mercury::core::IndexDistance::kMethodHalfFloatInnerProduct);
        } else {
            LOG_ERROR("Not supported method[%d] for half", half_index_meta.type());
            return false;
        }
        dist_scorer_factory_.Init(half_index_meta);
    }
    else {
        dist_scorer_factory_.Init(index_->GetIndexMeta());
    }

    if (!vector_retriever_.isValid() && !index_->ContainFeature()) {
        LOG_ERROR("can not found vector in array profile or attribute");
        return -1;
    }
    
    /*// for debug
    if (vector_retriever_.isValid()) {
        const void* addr_0 = nullptr;
        vector_retriever_(0, addr_0);
        const void* addr_1 = nullptr;
        vector_retriever_(1, addr_1);
        if (addr_0 != nullptr && addr_1 != nullptr) {
            LOG_INFO("addr_0 [%p]", addr_0);
            LOG_INFO("addr_1 [%p]", addr_1);
            LOG_INFO("offset [%ld]", (const char*)addr_1 - (const char*)addr_0);
        }
    }*/
    
    
    return 0;
}

void GroupIvfSearcher::SetIndex(Index::Pointer index)
{
    index_ = std::dynamic_pointer_cast<GroupIvfIndex>(index);
    if (index_->ForceHalf()) {
        auto half_index_meta = index_->GetIndexMeta();
        half_index_meta.setType(IndexMeta::kTypeHalfFloat);
        if (half_index_meta.method() == mercury::core::IndexDistance::kMethodFloatSquaredEuclidean) {
            half_index_meta.setMethod(mercury::core::IndexDistance::kMethodHalfFloatSquaredEuclidean);
        } else if (half_index_meta.method() == mercury::core::IndexDistance::kMethodFloatInnerProduct) {
            half_index_meta.setMethod(mercury::core::IndexDistance::kMethodHalfFloatInnerProduct);
        } else {
            LOG_ERROR("Not supported method[%d] for half", half_index_meta.type());
            return;
        }
        dist_scorer_factory_.Init(half_index_meta);
    }
    else {
        dist_scorer_factory_.Init(index_->GetIndexMeta());
    }
    index_->SetInMem(true);
}

void GroupIvfSearcher::SetBaseDocId(exdocid_t baseDocId)
{
    base_docid_ = baseDocId;
    index_->SetBaseDocid(baseDocId);
}

void GroupIvfSearcher::SetDeletionMapRetriever(const DeletionMapRetriever& retriever) {
    deletion_map_retriever_ = retriever;
    if (delmap_before) {
        index_->SetDeletionMapRetriever(retriever);
    }
}

IndexMeta::FeatureTypes GroupIvfSearcher::getFType()
{
    return this->index_->GetIndexMeta().type();
}

//! search by query
int GroupIvfSearcher::Search(const QueryInfo& query_info, GeneralSearchContext* context) {
    if (query_info.GetDimension() != index_->GetIndexMeta().dimension()) {
        LOG_ERROR("query dimension %lu != index dimension %lu.", query_info.GetDimension(), index_->GetIndexMeta().dimension());
        return -1;
    }

    const std::vector<GroupInfo>& group_infos = query_info.GetGroupInfos();

    if (query_info.MultiQueryMode()) {
        if (query_info.GetVectors().size() != group_infos.size()) {
            LOG_ERROR("num of query vector is not equal to group, %lu != %lu",
                      query_info.GetVectors().size(),
                      group_infos.size());
            return false;
        }
    }

    //定时器
    butil::Timer timer;

    //1. 保存n个group的结果堆，大小是total. 如果没有total就是topk。
    //1.1 分别对每个group调用search_ivf, 将所有postingIterator划分成P(并发数)份
    //1.2 对每个划分并行做DistScore计算，score结果放到临时vector中
    //1.3 所有score完成后，再分别把DistScore结果push到每个堆中。

    //2. 先从每个堆pop topk个，组成结果集1
    //3. 如果有total, 再从每个堆中余下的元素放到新的堆中，大小为total - sum(topk)
    //4. 最后total个元素按docid排序，去重返回
    std::vector<std::vector<off_t>>& real_slot_indexs = context->getAllGroupRealSlotIndexs();
    std::vector<uint32_t> group_doc_nums(group_infos.size(), 0); // 每个group有多少个doc
    std::vector<CoarseIndex<SmallBlock>::PostingIterator> ivf_postings;
    std::vector<uint32_t> ivf_postings_group_ids;
    MONITOR_TRANSACTION_LOG(CollectIvfPostings(ivf_postings, ivf_postings_group_ids, group_doc_nums, query_info, real_slot_indexs),
            "collect ivf posting failed.", 
            GroupIvf,
            CollectIvfPostings)

    MONITOR_METRIC_LOG(GroupIvf_CentroidNum, ivf_postings.size());
    MONITOR_METRIC_LOG(GroupIvf_GroupNum, group_infos.size());
    
    std::vector<DistNode> dist_nodes;
    MONITOR_TRANSACTION_LOG(CalcNodeDist(ivf_postings, ivf_postings_group_ids, query_info, dist_nodes, group_doc_nums), 
                "calc node dist failed.",
                GroupIvf,
                CalcNodeDist)

    if (index_->MultiAgeMode() && query_info.MultiAgeMode()) {
        MONITOR_TRANSACTION_LOG(CollectMultiAgeResult(query_info, dist_nodes, context),
                "collect multi age result failed.",
                GroupIvf,
                CollectMultiAgeResult);
        MONITOR_TRANSACTION_LOG(PostProcess(context, 1),
                "post process result failed.",
                GroupIvf,
                PostProcess);
        return 0;
    } else if (!index_->MultiAgeMode() && query_info.MultiAgeMode()) {
        LOG_WARN("index[%s] is not multi age mode, but query is, will recall from 0:0 group", index_name_.c_str());
    }
    
    if (query_info.GetGroupInfos().size() == 1) {
        uint32_t total = query_info.GetTotalRecall();
        if (total == 0 && !query_info.GetTopks().empty()) {
            total = query_info.GetTopks()[0];
        }
        MONITOR_TRANSACTION_LOG(CollectNthTopk(0, dist_nodes.size(), total, dist_nodes, context, 0),
                "collect nth topk failed.",
                GroupIvf,
                CollectNthTopk);
        MONITOR_TRANSACTION_LOG(PostProcess(context, group_infos.size()),
                "post process result failed.",
                GroupIvf,
                PostProcess);
        return 0;
    }

    MONITOR_TRANSACTION_LOG(CollectGroupHeaps(query_info, dist_nodes, group_doc_nums, context), 
                "collect group heaps failed.",
                GroupIvf,
                CollectGroupHeaps)

    // if (!query_info.GetContextParams().has(PARAM_GENERAL_RECALL_TEST_MODE)) {
    //     MONITOR_TRANSACTION_LOG(CollectBasicResult(group_heaps, query_info.GetTopks(), context), 
    //             "collect basic group heaps failed.",
    //             GroupIvf,
    //             CollectBasicResult)
    // }

    // MONITOR_TRANSACTION_LOG(CollectLeftResult(query_info.GetTotalRecall(), heap, context), 
    //             "collect left result failed.",
    //             GroupIvf,
    //             CollectLeftResult)

    MONITOR_TRANSACTION_LOG(PostProcess(context, group_infos.size()),
                "post process result failed.",
                GroupIvf,
                PostProcess);

    return 0;
}

int GroupIvfSearcher::CollectMultiAgeResult(const QueryInfo& query_info,
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
        // std::cout << "left: " << left << " right: " << right << std::endl;
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
        // std::cout << "start: " << start << " end: " << end << std::endl;
        // 实时内的2小时先做全排序后判断是否被删除，以保证quota
        bool full_sort = (age_infos[i].age <= 7200 && base_docid_ != 0);
        CollectNthTopk(start, end, topk, dist_nodes, context, i, full_sort);
    }
    timer.stop();
    transaction_SortInEachAge(timer.u_elapsed(), true);
    return 0;
}

int GroupIvfSearcher::CalcNodeDist(std::vector<CoarseIndex<SmallBlock>::PostingIterator>& ivf_postings,
                                   std::vector<uint32_t>& ivf_postings_group_ids,
                                   const QueryInfo& query_info,
                                   std::vector<DistNode>& dist_nodes,
                                   std::vector<uint32_t>& group_doc_nums) {
    size_t node_count = GetPostingsNodeCount(ivf_postings, 0, ivf_postings.size());
    if (base_docid_ == 0) {
        MONITOR_METRIC_LOG(GroupIvf_FullDocNum, node_count);
    }
    else {
        MONITOR_METRIC_LOG(GroupIvf_RtDocNum, node_count);
    }
    dist_nodes.reserve(node_count);

    //std::vector<size_t> ivf_postings_docids;
    if (delmap_before) {
        for (size_t i = 0; i < ivf_postings.size(); i++) {
            CoarseIndex<SmallBlock>::PostingIterator& iter = ivf_postings.at(i);
            while (UNLIKELY(!iter.finish())) {
                docid_t docid = iter.next();
                // offset means query vector offset
                if (deletion_map_retriever_.isValid() && deletion_map_retriever_(docid + base_docid_)) {
                    group_doc_nums[ivf_postings_group_ids.at(i)]--;
                    node_count--;
                    continue;
                }
                dist_nodes.emplace_back(docid, 0, ivf_postings_group_ids.at(i));
            }
        }
    }
    else {
        for (size_t i = 0; i < ivf_postings.size(); i++) {
            CoarseIndex<SmallBlock>::PostingIterator& iter = ivf_postings.at(i);
            while (UNLIKELY(!iter.finish())) {
                docid_t docid = iter.next();
                // offset means query vector offset
                dist_nodes.emplace_back(docid, 0, ivf_postings_group_ids.at(i));
            }
        }
    }

    PartitionStrategy strategy(node_count, mercury_doc_num_per_concurrency, mercury_max_concurrency_num, mercury_need_parallel);
    PartitionStrategy::return_type partition_result;
    if (strategy.MixedPartitionByDoc(node_count, partition_result) != 0) {
        LOG_ERROR("concurrency partition failed");
        return -1;
    }

    QueryInfo half_query_info;
    if (index_->ForceHalf()) {
        half_query_info.SetQuery(const_cast<QueryInfo&>(query_info).GetRawQuery());
        half_query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
        if (!half_query_info.MakeAsSearcher()) {
            LOG_ERROR("half_query_info make as searcher failed, %s", half_query_info.GetRawQuery().c_str());
            return -1;
        }
    }

    std::vector<bthread_t> bthreads;
    std::vector<SearcherMessage> msgs;
    msgs.resize(partition_result.size());
    for (size_t i = 0; i < partition_result.size() - 1; i++) {
        //先直接改为启动线程
        bthread_t bid;
        SearcherMessage& message = msgs.at(i);
        message.searcher = this;
        message.dist_nodes = &dist_nodes;
        message.node_start = partition_result[i].node_start_;
        message.node_end = partition_result[i].posting_end_;
        message.query_info = index_->ForceHalf() ? &half_query_info : &query_info;
        if (bthread_start_background(&bid, NULL, BthreadRun, &msgs.at(i)) != 0) {
            LOG_ERROR("start bthread failed.");
            return -1;
        }

        bthreads.push_back(bid);
    }

    //当前线程也进行计算。
    Partition& last_partition = partition_result[partition_result.size() - 1];
    BatchScore(last_partition.node_start_, last_partition.posting_end_, dist_nodes, index_->ForceHalf() ? half_query_info : query_info);

    for (auto t : bthreads) {
        bthread_join(t, NULL);
    }

    return 0;
}

int GroupIvfSearcher::CollectIvfPostings(std::vector<CoarseIndex<SmallBlock>::PostingIterator>& ivf_postings,
                                         std::vector<uint32_t>& ivf_postings_group_ids,
                                         std::vector<uint32_t>& group_doc_nums,
                                         const QueryInfo& query_info,
                                         std::vector<std::vector<off_t>>& real_slot_indexs) {
    const std::vector<GroupInfo>& group_infos = query_info.GetGroupInfos();
    bool is_rt = !real_slot_indexs.empty();
    bool is_multi_age = query_info.MultiAgeMode();
    bool is_recall_mode = query_info.GetContextParams().has(PARAM_GENERAL_RECALL_TEST_MODE);
    bool need_truncate = (is_rt && !is_recall_mode);
    
    if (is_rt && is_multi_age && index_->GetIndexParams().has(PARAM_RT_COARSE_SCAN_RATIO)) {
        real_slot_indexs.clear();
    }

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
                                  query_info.GetContextParams(), real_slot_indexs[i], is_rt, is_multi_age) != 0) {
                LOG_ERROR("Failed to call SearchIvf.");
                return -1;
            }
        }
    }
    index_->RecoverPostingFromSlot(ivf_postings, ivf_postings_group_ids, real_slot_indexs, group_doc_nums, need_truncate, query_info.MultiQueryMode());
    return 0;
}

int GroupIvfSearcher::CollectNthTopk(uint32_t start, uint32_t end, uint32_t topk,
                                     std::vector<DistNode>& dist_nodes,
                                     GeneralSearchContext* context,
                                     uint32_t offset, bool full_sort) const {
    if (end > start + topk) {
        if (full_sort) {
            std::sort(dist_nodes.begin() + start, dist_nodes.begin() + end);
        }
        else {
            std::nth_element(dist_nodes.begin() + start,
                            dist_nodes.begin() + start + topk,
                            dist_nodes.begin() + end);
        }
    }

    context->Result().reserve(context->Result().size() + topk);
    bool with_pk = index_->WithPk();
    uint32_t valid_topk = 0;
    for (size_t i = start; (i < start + topk || (full_sort && valid_topk < topk)) && i < end; i++) {
        const DistNode& node = dist_nodes[i];
        pk_t pk = 0;
        if (unlikely(with_pk)) {
            pk = index_->GetPk(node.key);
        }
        docid_t glo_doc_id = node.key + base_docid_;
        // TODO: move to before sort
        if (!delmap_before) {
            if (deletion_map_retriever_.isValid() && deletion_map_retriever_(glo_doc_id)) {
                continue;
            }
        }
        context->emplace_back(pk, glo_doc_id, node.dist, offset);
        valid_topk++;
    }
    
    return 0;
}

// TODO
int GroupIvfSearcher::CollectGroupHeaps(const QueryInfo& query_info,
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
            if (!delmap_before) {
                if (deletion_map_retriever_.isValid() && deletion_map_retriever_(glo_doc_id)) {
                    continue;
                }
            }
            context->emplace_back(0, glo_doc_id, node.dist, i);
        }
        group_start += group_doc_num;
    }
    return 0;
}

int GroupIvfSearcher::CollectBasicResult(std::vector<MyHeap<DistNode>>& group_heaps, const std::vector<uint32_t>& topks,
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
            if (!delmap_before) {
                if (deletion_map_retriever_.isValid() && deletion_map_retriever_(glo_doc_id)) {
                    continue;
                }
            }
            context->emplace_back(pk, glo_doc_id, node->dist);
        }
    }

    return 0;
}

int GroupIvfSearcher::CollectLeftResult(uint32_t total, std::vector<MyHeap<DistNode>>& group_heaps,
                                        GeneralSearchContext* context) const {
    //如果有total, 再从每个堆中余下的元素放到新的堆中，大小为total - sum(topk)
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

int GroupIvfSearcher::PostProcess(GeneralSearchContext* context, size_t group_num) const {
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

    return 0;
}


void GroupIvfSearcher::PushSearchResultToContext(GeneralSearchContext* context, const std::vector<DistNode>& dist_vec) const {
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


void PushToHeap(MyHeap<DistNode>& dist_heap, MyHeap<DistNode>& left_result) {
    while(!dist_heap.fetchEnd()) {
        const DistNode& node = *dist_heap.fetch();
        left_result.push(std::move(node));
    }
}

MERCURY_NAMESPACE_END(core);
