#include "ivf_fast_scan_searcher.h"
#include "src/core/framework/utility/closure.h"
#include "src/core/algorithm/thread_common.h"
#include "bthread/bthread.h"
#include <chrono>

MERCURY_NAMESPACE_BEGIN(core);

void IvfFastScanSearcher::BatchScore(size_t node_start, size_t node_end,
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


size_t IvfFastScanSearcher::GetPostingsNodeCount(const std::vector<CoarseIndex<BigBlock>::PostingIterator>& ivf_postings,
                                              size_t start, size_t end) const {
    size_t count = 0;
    for (size_t i = start; i < end; i++) {
        count += ivf_postings.at(i).getDocNum();
    }
    return count;
}

IvfFastScanSearcher::IvfFastScanSearcher() {
    if (!index_) {
        index_.reset(new IvfFastScanIndex());
    }
    SetThreadEnv();
}

int IvfFastScanSearcher::Init(IndexParams &params) {
    index_->SetIndexParams(params);

    std::string index_name = params.getString(PARAM_VECTOR_INDEX_NAME);
    LOG_INFO("Start Init IvfFastScanSearcher, %s", index_name.c_str());

    MONITOR_TRANSACTION(IvfFastScan, CollectPostsNPackers);
    MONITOR_TRANSACTION(IvfFastScan, FastScanNodeDist);
    MONITOR_TRANSACTION(IvfFastScan, CollectGroupHeaps);
    MONITOR_TRANSACTION(IvfFastScan, CollectBasicResult);
    MONITOR_TRANSACTION(IvfFastScan, CollectLeftResult);
    MONITOR_TRANSACTION(IvfFastScan, CollectNthTopk);
    MONITOR_TRANSACTION(IvfFastScan, PostProcess);
    MONITOR_TRANSACTION(IvfFastScan, CollectMultiAgeResult);
    MONITOR_TRANSACTION(IvfFastScan, StatAgeInfo);
    MONITOR_TRANSACTION(IvfFastScan, GenerateAgeSortedContainer);
    MONITOR_TRANSACTION(IvfFastScan, SortInEachAge);
    MONITOR_METRIC_WITH_INDEX(IvfFastScan_CentroidNum, "IvfFastScan_CentroidNum_" + index_name);
    MONITOR_METRIC_WITH_INDEX(IvfFastScan_GroupNum, "IvfFastScan_GroupNum_" + index_name);
    MONITOR_METRIC_WITH_INDEX(IvfFastScan_DocNum, "IvfFastScan_DocNum_" + index_name);

    return 0;
}

int IvfFastScanSearcher::LoadIndex(const std::string& path) {
    //TODO
    return -1;
}

int IvfFastScanSearcher::LoadIndex(const void* data, size_t size) {
    if (index_->Load(data, size) != 0) {
        LOG_ERROR("Failed to load index.");
        return -1;
    }
    dist_scorer_factory_.Init(index_->GetIndexMeta());
    return 0;
}

void IvfFastScanSearcher::SetIndex(Index::Pointer index)
{
    index_ = std::dynamic_pointer_cast<IvfFastScanIndex>(index);
    dist_scorer_factory_.Init(index_->GetIndexMeta());
}

void IvfFastScanSearcher::SetBaseDocId(exdocid_t baseDocId)
{
    base_docid_ = baseDocId;
    index_->SetBaseDocid(baseDocId);
}

IndexMeta::FeatureTypes IvfFastScanSearcher::getFType()
{
    return this->index_->GetIndexMeta().type();
}

int IvfFastScanSearcher::Search(const QueryInfo& query_info, GeneralSearchContext* context) {

    const std::vector<GroupInfo>& group_infos = query_info.GetGroupInfos();

    if (query_info.MultiQueryMode()) {
        LOG_ERROR("no multi-query mode support for fastscan");
        return false;
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
    std::vector<FastScanIndex::PackedCodeIterator> code_packers;
    std::vector<uint32_t> ivf_postings_group_ids;
    MONITOR_TRANSACTION_LOG(CollectPostsNPackers(ivf_postings, code_packers, ivf_postings_group_ids,
                                                group_doc_nums, query_info, real_slot_indexs), 
                                                "collect ivf posting failed.", 
                                                IvfFastScan,
                                                CollectPostsNPackers)

    MONITOR_METRIC_LOG(IvfFastScan_CentroidNum, ivf_postings.size());
    MONITOR_METRIC_LOG(IvfFastScan_GroupNum, group_infos.size());

    std::vector<DistNode> dist_nodes;
    MONITOR_TRANSACTION_LOG(FastScanNodeDist(ivf_postings, code_packers, ivf_postings_group_ids, query_info, dist_nodes), 
                "fastscan node dist failed.",
                IvfFastScan,
                FastScanNodeDist)

    if (query_info.GetGroupInfos().size() == 1) {
        uint32_t total = query_info.GetTotalRecall();
        if (total == 0 && !query_info.GetTopks().empty()) {
            total = query_info.GetTopks()[0];
        }
        MONITOR_TRANSACTION_LOG(CollectNthTopk(0, dist_nodes.size(), total, dist_nodes, context, 0),
                "collect nth topk failed.",
                IvfFastScan,
                CollectNthTopk);
        MONITOR_TRANSACTION_LOG(PostProcess(query_info, context, group_infos.size()),
                "post process result failed.",
                IvfFastScan,
                PostProcess);
        return 0;
    }

    MONITOR_TRANSACTION_LOG(CollectGroupHeaps(query_info, dist_nodes, group_doc_nums, context), 
                "collect group heaps failed.",
                IvfFastScan,
                CollectGroupHeaps)

    MONITOR_TRANSACTION_LOG(PostProcess(query_info, context, group_infos.size()),
                "post process result failed.",
                IvfFastScan,
                PostProcess);

    return 0;
}

int IvfFastScanSearcher::CollectNthTopk(uint32_t start, uint32_t end, uint32_t topk,
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

int IvfFastScanSearcher::FastScanNodeDist(std::vector<CoarseIndex<BigBlock>::PostingIterator>& ivf_postings,
                                        std::vector<FastScanIndex::PackedCodeIterator>& code_packers,
                                        std::vector<uint32_t>& ivf_postings_group_ids,
                                        const QueryInfo& query_info, 
                                        std::vector<DistNode>& dist_nodes) {
    if (ivf_postings.size() == 0) {
        LOG_WARN("ivf_postings size is zero");
        return 0;
    }
    size_t total_node_count = GetPostingsNodeCount(ivf_postings, 0, ivf_postings.size());
    MONITOR_METRIC_LOG(IvfFastScan_DocNum, total_node_count);
    dist_nodes.reserve(total_node_count);

    bool is_recall_mode = false;
    // 测召回模式，全走normal距离
    if (UNLIKELY(query_info.GetContextParams().has(PARAM_GENERAL_RECALL_TEST_MODE) &&
        query_info.GetContextParams().has(PARAM_COARSE_SCAN_RATIO))) {
        auto ratio = query_info.GetContextParams().getFloat(PARAM_COARSE_SCAN_RATIO);
        if (std::abs(ratio - 1.0) < 1e-6) {
            is_recall_mode = true;
        }
    }

    assert(ivf_postings.size() == code_packers.size());

    // init LUT
    LoopUpTable* lut = new LoopUpTable(index_->GetIndexMeta(), &(index_->GetPqCentroidResource()));
    if (!lut->initLoopUpTable(query_info.GetVectors().at(0))) {
        LOG_ERROR("Init lookup table error");
        return -1;
    }

    // quantize LUT
    if (!lut->quantizeLoopUpTable()) {
        LOG_ERROR("Failed to quantize distance matrix for fastscan");
    }

    const uint8_t *quantizedLut = 
        reinterpret_cast<const uint8_t *>(lut->GetQDistanceArray());
    size_t capacity = lut->getFragmentNum() * lut->getCentroidNum();
    uint8_t* packedLut = reinterpret_cast<uint8_t*>(aligned_alloc(32, capacity));
    for (size_t i = 0; i < lut->getFragmentNum(); i++) {
        for (size_t j = 0; j < lut->getCentroidNum(); ++j) {
            packedLut[i * lut->getCentroidNum() + j] = quantizedLut[i * lut->getCentroidNum() + j];
        }
    }

    size_t doc_offset = 0;

    for (size_t i = 0; i < ivf_postings.size(); i++) {
        CoarseIndex<BigBlock>::PostingIterator& ivf_posting = ivf_postings.at(i);
        if (ivf_posting.getDocNum() == 0) {
            LOG_WARN("ivf_posting[%lu] has no doc for search", i);
        } else {
            while (UNLIKELY(!ivf_posting.finish())) {
                docid_t docid = ivf_posting.next();
                dist_nodes.emplace_back(docid, 0, ivf_postings_group_ids.at(i));
            }
            if (!is_recall_mode) {
                FastScanIndex::PackedCodeIterator& code_packer = code_packers.at(i);
                if (code_packer.getCodeCount() == 0) {
                    LOG_ERROR("code_packer[%lu] has no code for search", i);
                    return -1;
                } else {
                    FastScanScorer scorer;
                    scorer.score(packedLut, code_packer.getStartPointer(),
                                lut->getFragmentNum(), lut->getScale(), lut->getBias(),
                                doc_offset, ivf_posting.getDocNum(), &dist_nodes);
                    doc_offset += ivf_posting.getDocNum();
                }
            }
        }
    }

    if (is_recall_mode) {
        BatchScore(0, total_node_count - 1, &dist_nodes, query_info);
    }

    delete lut;

    free(packedLut);

    return 0;
}

int IvfFastScanSearcher::CollectPostsNPackers(std::vector<CoarseIndex<BigBlock>::PostingIterator>& ivf_postings,
                                            std::vector<FastScanIndex::PackedCodeIterator>& code_packers,
                                            std::vector<uint32_t>& ivf_postings_group_ids,
                                            std::vector<uint32_t>& group_doc_nums,
                                            const QueryInfo& query_info,
                                            std::vector<std::vector<off_t>>& real_slot_indexs) {
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
                                  query_info.GetContextParams(), real_slot_indexs[i]) != 0) {
                LOG_ERROR("Failed to call SearchIvf.");
                return -1;
            }
        }
    }
    index_->GetPostNPackerFromSlot(ivf_postings, code_packers, ivf_postings_group_ids,
                                real_slot_indexs, group_doc_nums, query_info.MultiQueryMode());
    return 0;
}

// TODO: ensure group quota
int IvfFastScanSearcher::CollectGroupHeaps(const QueryInfo& query_info,
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

int IvfFastScanSearcher::CollectBasicResult(std::vector<MyHeap<DistNode>>& group_heaps, const std::vector<uint32_t>& topks,
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

int IvfFastScanSearcher::CollectLeftResult(uint32_t total, std::vector<MyHeap<DistNode>>& group_heaps,
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

int IvfFastScanSearcher::PostProcess(const QueryInfo& query_info, GeneralSearchContext* context, 
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


void IvfFastScanSearcher::PushSearchResultToContext(GeneralSearchContext* context, const std::vector<DistNode>& dist_vec) const {
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
