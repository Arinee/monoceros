#pragma once

#include <memory>
#include <string>
#include <vector>
#include "src/core/framework/index_framework.h"
#include "src/core/algorithm/general_search_context.h"
#include "src/core/algorithm/searcher.h"
#include "fastscan_scorer.h"
#include "src/core/algorithm/orig_dist_scorer.h"
#include "src/core/algorithm/partition_strategy.h"
#include "src/core/utils/my_heap.h"
#include "src/core/utils/monitor_component.h"
#include "src/core/framework/utility/thread_pool.h"
#include "ivf_fast_scan_index.h"
#include "src/core/utils/monitor_component.h"

MERCURY_NAMESPACE_BEGIN(core);

void PushToHeap(MyHeap<DistNode>& dist_heap, MyHeap<DistNode>& left_result);

class IvfFastScanSearcher : public Searcher
{
public:
    typedef std::shared_ptr<IvfFastScanSearcher> Pointer;
public:
    IvfFastScanSearcher();
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
    transaction transaction_CollectPostsNPackers, \
                transaction_FastScanNodeDist, transaction_CollectNthTopk, \
                transaction_CollectGroupHeaps, transaction_CollectBasicResult, \
                transaction_CollectLeftResult, transaction_PostProcess, \
                transaction_CollectMultiAgeResult, transaction_StatAgeInfo,     \
                transaction_GenerateAgeSortedContainer, transaction_SortInEachAge;
    //监控访问中心点数centroid_num，类目数group_num，文档数doc_num
    metric metric_IvfFastScan_CentroidNum, metric_IvfFastScan_GroupNum, metric_IvfFastScan_DocNum;
public:
    int SearchIvf(std::vector<CoarseIndex<BigBlock>::PostingIterator>& postings, const void* data, size_t size);

private:
    void PushSearchResultToContext(GeneralSearchContext* context, const std::vector<DistNode>& dist_vec) const;

    int FastScanNodeDist(std::vector<CoarseIndex<BigBlock>::PostingIterator>& ivf_postings,
                        std::vector<FastScanIndex::PackedCodeIterator>& code_packers,
                        std::vector<uint32_t>& ivf_postings_group_ids,
                        const QueryInfo& query_info, std::vector<DistNode>& dist_nodes);

    int CollectPostsNPackers(std::vector<CoarseIndex<BigBlock>::PostingIterator>& ivf_postings,
                            std::vector<FastScanIndex::PackedCodeIterator>& code_packers,
                            std::vector<uint32_t>& ivf_postings_group_ids,
                            std::vector<uint32_t>& group_doc_nums,
                            const QueryInfo& query_info,
                            std::vector<std::vector<off_t>>& real_slot_indexs);

    int CollectGroupHeaps(const QueryInfo& query_info,
                          std::vector<DistNode>& dist_nodes,
                          const std::vector<uint32_t>& group_posting_nums,
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

    size_t GetPostingsNodeCount(const std::vector<CoarseIndex<BigBlock>::PostingIterator>& ivf_postings,
                                size_t start, size_t end) const;

private:
    IvfFastScanIndex::Pointer index_;
    OrigDistScorer::Factory dist_scorer_factory_;
};

MERCURY_NAMESPACE_END(core);
