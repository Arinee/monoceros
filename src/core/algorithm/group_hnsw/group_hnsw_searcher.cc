#include "group_hnsw_searcher.h"
#include "bthread/bthread.h"
#include <errno.h>
#include <pthread.h>

MERCURY_NAMESPACE_BEGIN(core);

GroupHnswSearcher::GroupHnswSearcher()
{
    if (!index_) {
        index_.reset(new GroupHnswIndex());
    }
};

GroupHnswSearcher::~GroupHnswSearcher(){};

int GroupHnswSearcher::Init(IndexParams &params)
{

    index_->SetIndexParams(params);
    params_ = params;

    part_dimension_ = params.getUint64(PARAM_CUSTOMED_PART_DIMENSION);

    std::string index_name = params.getString(PARAM_VECTOR_INDEX_NAME);
    LOG_INFO("Start Init GroupHnswSearcher, %s", index_name.c_str());

    MONITOR_METRIC(GroupHnsw_GroupNum);
    MONITOR_METRIC_WITH_INDEX(GroupHnsw_BruteCmpCnt, "GroupHnsw_BruteCompareCnt_" + index_name);
    MONITOR_METRIC_WITH_INDEX(GroupHnsw_HnswCmpCnt, "GroupHnsw_HnswCmpCnt_" + index_name);

    LOG_INFO("End Init GroupHnswSearcher, %s", index_name.c_str());

    return 0;
}

int GroupHnswSearcher::LoadIndex(const std::string &path)
{
    // no use
    return 0;
}
int GroupHnswSearcher::LoadIndex(const void *data, size_t size)
{
    if (index_->Load(data, size) != 0) {
        LOG_ERROR("Load coarse_hnsw_index failed in searcher");
        return -1;
    }

    if (!vector_retriever_.isValid() && !index_->ContainFeature()) {
        LOG_ERROR("can not found vector in array profile or attribute");
        return -1;
    }

    return 0;
}

void GroupHnswSearcher::SetIndex(Index::Pointer index)
{
    index_ = std::dynamic_pointer_cast<GroupHnswIndex>(index);
}

void GroupHnswSearcher::SetBaseDocId(exdocid_t baseDocId)
{
    base_docid_ = baseDocId;
    index_->SetBaseDocid(baseDocId);
}

void GroupHnswSearcher::SetVectorRetriever(const AttrRetriever &retriever)
{
    vector_retriever_ = retriever;
    index_->SetVectorRetriever(retriever);
}

IndexMeta::FeatureTypes GroupHnswSearcher::getFType()
{
    return this->index_->GetIndexMeta().type();
}

int GroupHnswSearcher::Search(const QueryInfo &query_info, GeneralSearchContext *context)
{
    // QueryInfo query_info(query_str);
    // if (index_->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeBinary) {
    //     query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeBinary);
    // }
    // if (!query_info.MakeAsSearcher()) {
    //     LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
    //     return -1;
    // }
    if (part_dimension_ == 0) {
        if (query_info.GetDimension() != index_->GetIndexMeta().dimension()) {
            LOG_ERROR("query dimension %lu != index dimension %lu.", query_info.GetDimension(),
                      index_->GetIndexMeta().dimension());
            return -1;
        }
    }

    if (query_info.MultiAgeMode()) {
        LOG_ERROR("Hnsw doesn't support multiage mode now.");
        return -1;
    }

    const std::vector<GroupInfo> &group_infos = query_info.GetGroupInfos();

    bool is_multi_query = false;
    if (query_info.GetContextParams().has(PARAM_MULTI_QUERY_MODE)) {
        is_multi_query = true;
        if (query_info.GetVectors().size() != group_infos.size()) {
            LOG_ERROR("num of query vector is not equal to group, %lu != %lu", query_info.GetVectors().size(),
                      group_infos.size());
            return false;
        }
    }

    const std::vector<uint32_t> &topks = query_info.GetTopks();
    const uint32_t total = query_info.GetTotalRecall();
    std::vector<MyHeap<DistNode>> group_heaps;
    group_heaps.reserve(group_infos.size());

    int max_scan_num_in_query = index_->GetCoarseHnswIndex().getMaxScanNums();
    if (query_info.GetContextParams().has(PARAM_GRAPH_MAX_SCAN_NUM_IN_QUERY)) {
        max_scan_num_in_query = query_info.GetContextParams().getInt32(PARAM_GRAPH_MAX_SCAN_NUM_IN_QUERY);
    }
    float downgrade_percent = query_info.GetContextParams().has(PARAM_DOWNGRADE_PERCENT)
                                  ? query_info.GetContextParams().getFloat(PARAM_DOWNGRADE_PERCENT)
                                  : 1;
    max_scan_num_in_query = max_scan_num_in_query * downgrade_percent;

    std::vector<std::pair<int, int>> cmp_cnts(group_infos.size());

    // for make groudtruth and cal recall rate
    if (query_info.GetContextParams().has(PARAM_GENERAL_RECALL_TEST_MODE) &&
        query_info.GetContextParams().has(PARAM_COARSE_SCAN_RATIO)) {
        auto ratio = query_info.GetContextParams().getFloat(PARAM_COARSE_SCAN_RATIO);
        if (std::abs(ratio - 1.0) < 1e-6) {
            for (size_t i = 0; i < group_infos.size(); i++) {
                index_->BruteSearch(group_infos.at(i), total,
                                    is_multi_query ? query_info.GetVectors().at(i) : query_info.GetVectors().at(0),
                                    query_info.GetVectorLen(), context, group_heaps, cmp_cnts.at(i));
            }
        } else {
            for (size_t i = 0; i < group_infos.size(); i++) {
                MyHeap<DistNode> group_heap(total);
                group_heaps.push_back(std::move(group_heap));
                index_->KnnSearch(group_infos.at(i), total,
                                  is_multi_query ? query_info.GetVectors().at(i) : query_info.GetVectors().at(0),
                                  query_info.GetVectorLen(), context, &group_heaps.at(i), max_scan_num_in_query,
                                  cmp_cnts.at(i));
            }
        }
        if (CollectLeftResult(total, group_heaps, context) != 0) {
            LOG_ERROR("collect left result failed.");
            return -1;
        }

        PostProcess(context, group_infos.size());
        return 0;
    }

    std::vector<bthread_t> bthreads;
    std::vector<SearcherMessage> msgs;
    msgs.resize(group_infos.size() - 1);

    size_t i = 0;
    for (; i < group_infos.size() - 1; i++) {

        bthread_t bid;
        SearcherMessage &message = msgs.at(i);
        message.index = index_.get();
        message.group_info = &group_infos.at(i);
        message.topk = topks.at(i);
        if (message.topk == 0) {
            message.topk = total;
        }
        MyHeap<DistNode> group_heap(message.topk);
        group_heaps.push_back(std::move(group_heap));
        message.query_vector = is_multi_query ? query_info.GetVectors().at(i) : query_info.GetVectors().at(0);
        message.vector_length = query_info.GetVectorLen();
        message.context = context;
        message.group_heap = &group_heaps.at(i);
        message.max_scan_num_in_query = max_scan_num_in_query;
        message.cmp_cnt = &cmp_cnts.at(i);
        if (bthread_start_background(&bid, NULL, BthreadRun, &msgs.at(i)) != 0) {
            LOG_ERROR("start bthread failed.");
            return -1;
        }

        bthreads.push_back(bid);
    }

    // 当前线程也进行计算
    uint32_t topk = topks.at(i);
    if (topk == 0) {
        topk = total;
    }
    MyHeap<DistNode> group_heap(topk);
    group_heaps.push_back(std::move(group_heap));
    index_->KnnSearch(group_infos.at(i), topk,
                      is_multi_query ? query_info.GetVectors().at(i) : query_info.GetVectors().at(0),
                      query_info.GetVectorLen(), context, &group_heaps.at(i), max_scan_num_in_query, cmp_cnts.at(i));

    for (auto t : bthreads) {
        bthread_join(t, NULL);
    }

    uint32_t brute_cmp_cnt = 0, hnsw_cmp_cnt = 0;
    for (auto &cmp_cnt : cmp_cnts) {
        brute_cmp_cnt += cmp_cnt.first;
        hnsw_cmp_cnt += cmp_cnt.second;
    }
    MONITOR_METRIC_LOG(GroupHnsw_GroupNum, group_infos.size());
    MONITOR_METRIC_LOG(GroupHnsw_BruteCmpCnt, brute_cmp_cnt);
    MONITOR_METRIC_LOG(GroupHnsw_HnswCmpCnt, hnsw_cmp_cnt);

    if (CollectGroupResults(query_info, group_heaps, context) != 0) {
        LOG_ERROR("collect group results failed.");
        return -1;
    }

    // if (CollectBasicResult(group_heaps, query_info.GetTopks(), context) != 0) {
    //     LOG_ERROR("collect basic group heaps failed.");
    //     return -1;
    // }

    // if (CollectLeftResult(total, group_heaps, context) != 0) {
    //     LOG_ERROR("collect left result failed.");
    //     return -1;
    // }

    PostProcess(context, group_infos.size());
    return 0;
}

int GroupHnswSearcher::CollectGroupResults(const QueryInfo &query_info, std::vector<MyHeap<DistNode>> &group_heaps,
                                           GeneralSearchContext *context) const
{
    uint32_t total = query_info.GetTotalRecall();
    if (total == 0) {
        for (size_t i = 0; i < query_info.GetTopks().size(); i++) {
            total += query_info.GetTopks()[i];
        }
    }
    context->Result().reserve(total);

    for (size_t i = 0; i < group_heaps.size(); i++) {
        MyHeap<DistNode> &result = group_heaps[i];
        uint32_t topk = query_info.GetTopks().at(i);
        if (topk == 0) {
            topk = total;
        }
        for (size_t j = 0; j < topk && !result.fetchEnd(); j++) {
            const DistNode *node = result.fetch();
            if (node == nullptr) {
                LOG_ERROR("fetch node null.");
                return -1;
            }
            docid_t glo_doc_id = node->key + base_docid_;
            if (deletion_map_retriever_.isValid() && deletion_map_retriever_(glo_doc_id)) {
                continue;
            }
            context->emplace_back(0, glo_doc_id, node->dist, i);
        }
    }
    return 0;
}

int GroupHnswSearcher::CollectBasicResult(std::vector<MyHeap<DistNode>> &group_heaps,
                                          const std::vector<uint32_t> &topks, GeneralSearchContext *context) const
{
    // 先从每个堆pop topk个，组成结果集1
    bool with_pk = index_->WithPk();
    size_t capacity = 0;
    for (size_t i = 0; i < group_heaps.size(); i++) {
        capacity += topks.at(i);
    }
    context->Result().reserve(capacity);

    for (size_t i = 0; i < group_heaps.size(); i++) {
        MyHeap<DistNode> &result = group_heaps.at(i);
        uint32_t topk = topks.at(i);
        for (size_t j = 0; j < topk && !result.fetchEnd(); j++) {
            const DistNode *node = result.fetch();
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

int GroupHnswSearcher::CollectLeftResult(uint32_t total, std::vector<MyHeap<DistNode>> &group_heaps,
                                         GeneralSearchContext *context) const
{
    // 如果有total, 再从每个堆中余下的元素放到新的堆中，大小为total - sum(topk)
    int64_t left_count = total - context->Result().size();
    if (total > 0 && left_count > 0) {
        MyHeap<DistNode> left_result(left_count);
        for (size_t i = 0; i < group_heaps.size(); i++) {
            MyHeap<DistNode> &group_heap = group_heaps.at(i);
            while (!group_heap.fetchEnd()) {
                const DistNode &node = *group_heap.fetch();
                left_result.push(std::move(node));
            }
        }
        bool with_pk = index_->WithPk();
        context->Result().reserve(context->Result().size() + left_result.getData().size());
        for (size_t i = 0; i < left_result.getData().size(); i++) {
            const DistNode &node = left_result.getData().at(i);
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

    return 0;
}

int GroupHnswSearcher::PostProcess(GeneralSearchContext *context, size_t group_num) const
{
    std::vector<SearchResult> &results = context->Result();
    std::sort(results.begin(), results.end(),
              [](const SearchResult &a, const SearchResult &b) { return a.gloid < b.gloid; });
    // 如果多于一个分组，返回结果去重
    if (group_num > 1) {
        results.erase(std::unique(results.begin(), results.end()), results.end());
    }

    return 0;
}

void *GroupHnswSearcher::BthreadRun(void *message)
{
    SearcherMessage *msg = static_cast<SearcherMessage *>(message);
    if (msg->index) {
        msg->index->KnnSearch(*msg->group_info, msg->topk, msg->query_vector, msg->vector_length, msg->context,
                              msg->group_heap, msg->max_scan_num_in_query, *msg->cmp_cnt);
    }
    return nullptr;
}

MERCURY_NAMESPACE_END(core);
