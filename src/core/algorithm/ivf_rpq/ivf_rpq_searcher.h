# pragma once

#include <memory>
#include <string>
#include <vector>
#include "src/core/framework/index_framework.h"
#include "src/core/algorithm/general_search_context.h"
#include "src/core/algorithm/searcher.h"
#include "src/core/algorithm/ivf_rpq/pq_dist_scorer1.h"
#include "src/core/algorithm/orig_dist_scorer.h"
#include "src/core/algorithm/partition_strategy.h"
#include "src/core/utils/my_heap.h"
#include "src/core/utils/monitor_component.h"
#include "src/core/framework/utility/thread_pool.h"
#include "ivf_rpq_index.h"
#include "src/core/utils/monitor_component.h"

MERCURY_NAMESPACE_BEGIN(core);

void PushToHeap(MyHeap<DistNode>& dist_heap, MyHeap<DistNode>& left_result);

class IvfRpqSearcher : public Searcher
{
public:
    typedef std::shared_ptr<IvfRpqSearcher> Pointer;
public:
    IvfRpqSearcher();
    //! Init from params
    int Init(IndexParams &params) override;

    int LoadIndex(const std::string& path) override;
    int LoadIndex(const void* data, size_t size) override;
    void SetIndex(Index::Pointer index) override;
    void SetBaseDocId(exdocid_t baseDocId) override;
    IndexMeta::FeatureTypes getFType() override;
    //! search by query
    int Search(const QueryInfo& query_info, GeneralSearchContext* context = nullptr);

    typedef std::function<void(int /* latency_us */, bool /* succ */)> transaction;
    typedef std::function<void(int /* value */)> metric;

    //监控各操作延时
    transaction transaction_CollectIvfPostings, transaction_CalcNodeDist, \
                transaction_CollectNthTopk, \
                transaction_CollectGroupHeaps, transaction_CollectBasicResult, \
                transaction_CollectLeftResult, transaction_PostProcess, \
                transaction_CollectMultiAgeResult, transaction_StatAgeInfo,     \
                transaction_GenerateAgeSortedContainer, transaction_SortInEachAge;
    //监控访问中心点数centroid_num，类目数group_num，文档数doc_num
    metric metric_IvfRpq_CentroidNum, metric_IvfRpq_GroupNum, metric_IvfRpq_FullDocNum, metric_IvfRpq_RtDocNum;
public:
    int SearchIvf(std::vector<CoarseIndex<SmallBlock>::PostingIterator>& postings, const void* data, size_t size);

private:
    struct SearcherMessage {
        IvfRpqSearcher* searcher;
        //std::vector<size_t>* ivf_postings_docids;
        size_t node_start;
        //size_t posting_start;
        size_t node_end;
        std::vector<DistNode>* dist_nodes;
        const QueryInfo* query_info;
        // QueryDistanceMatrix1* qdm;
        std::vector<QueryDistanceMatrix1 *>* qdms;
    };
    static void* BthreadRun(void* message);

    void PushSearchResultToContext(GeneralSearchContext* context, const std::vector<DistNode>& dist_vec) const;

    int CalcNodeDist(std::vector<CoarseIndex<SmallBlock>::PostingIterator>& ivf_postings,
                        const QueryInfo& query_info, std::vector<DistNode>& dist_nodes,
                        std::vector<std::pair<uint32_t, distance_t>>& lvl1_idx_dis);

    int CollectIvfPostings(std::vector<CoarseIndex<SmallBlock>::PostingIterator>& ivf_postings,
                           std::vector<uint32_t>& ivf_postings_group_ids,
                           std::vector<uint32_t>& group_doc_nums,
                           const QueryInfo& query_info,
                           std::vector<std::vector<off_t>>& real_slot_indexs, 
                           std::vector<std::pair<uint32_t, distance_t>>& lvl1_idx_dis);

    int CollectGroupHeaps(const QueryInfo& query_info,
                          std::vector<DistNode>& dist_nodes,
                          const std::vector<uint32_t>& group_posting_nums,
                          GeneralSearchContext* context) const;
    
    int CollectMultiAgeResult(const QueryInfo& query_info,
                              std::vector<DistNode>& dist_nodes,
                              GeneralSearchContext* context) const;

    int CollectNthTopk(uint32_t start, uint32_t end, uint32_t topk,
                       std::vector<DistNode>& dist_nodes,
                       GeneralSearchContext* context,
                       uint32_t offset) const;

    int PostProcess(const QueryInfo& query_info, GeneralSearchContext* context, size_t group_num) const;

    int CollectBasicResult(std::vector<MyHeap<DistNode>>& group_heaps, const std::vector<uint32_t>& topks,
                          GeneralSearchContext* context) const;

    int CollectLeftResult(uint32_t total, std::vector<MyHeap<DistNode>>& group_heaps,
                          GeneralSearchContext* context) const;

    void BatchScore(size_t node_start, size_t node_end,
                    std::vector<DistNode>* dist_nodes,
                    const QueryInfo& query_info) const;

    void BatchScoreRpq(size_t node_start, size_t node_end,
                    std::vector<DistNode>* dist_nodes1,
                    QueryDistanceMatrix1* qdm) const;

    size_t GetPostingsNodeCount(const std::vector<CoarseIndex<SmallBlock>::PostingIterator>& ivf_postings,
                                size_t start, size_t end) const;

private:
    IvfRpqIndex::Pointer index_;
    OrigDistScorer::Factory dist_scorer_factory_;
};

MERCURY_NAMESPACE_END(core);
