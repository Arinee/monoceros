#ifndef __MERCURY_CORE_GROUP_IVF_SEARCHER_H__
#define __MERCURY_CORE_GROUP_IVF_SEARCHER_H__

#include <memory>
#include <string>
#include <vector>
#include "src/core/framework/index_framework.h"
#include "../general_search_context.h"
#include "../searcher.h"
#include "group_ivf_index.h"
#include "../orig_dist_scorer.h"
#include "src/core/utils/my_heap.h"
#include "src/core/framework/utility/thread_pool.h"
#include "../query_info.h"
#include "src/core/utils/monitor_component.h"

MERCURY_NAMESPACE_BEGIN(core);

void PushToHeap(MyHeap<DistNode>& dist_heap, MyHeap<DistNode>& left_result);

class GroupIvfSearcher : public Searcher
{
public:
    typedef std::shared_ptr<GroupIvfSearcher> Pointer;
public:
    GroupIvfSearcher();
    //! Init from params
    int Init(IndexParams &params) override;

    int LoadIndex(const std::string& path) override;
    int LoadIndex(const void* data, size_t size) override;
    void SetIndex(Index::Pointer index) override;
    void SetBaseDocId(exdocid_t baseDocId) override;
    void SetDeletionMapRetriever(const DeletionMapRetriever& retriever) override;
    IndexMeta::FeatureTypes getFType() override;
    //! search by query
    int Search(const QueryInfo& query_info, GeneralSearchContext* context = nullptr);

    typedef std::function<void(int /* latency_us */, bool /* succ */)> transaction;
    typedef std::function<void(int /* value */)> metric;
    //监控各操作延时
    transaction transaction_CollectIvfPostings, transaction_CalcNodeDist, \
                transaction_CollectNthTopk, transaction_CollectGroupHeaps, \
                transaction_CollectBasicResult, transaction_CollectLeftResult, \
                transaction_CollectMultiAgeResult, transaction_PostProcess, \
                transaction_StatAgeInfo, transaction_GenerateAgeSortedContainer,    \
                transaction_SortInEachAge;
    //监控访问中心点数centroid_num，类目数group_num，文档数doc_num
    metric metric_GroupIvf_CentroidNum, metric_GroupIvf_GroupNum, metric_GroupIvf_FullDocNum, metric_GroupIvf_RtDocNum;
public:
    int SearchIvf(std::vector<CoarseIndex<SmallBlock>::PostingIterator>& postings, const void* data, size_t size, bool is_multi_query);

private:
    struct SearcherMessage {
        GroupIvfSearcher* searcher;
        size_t node_start;
        size_t node_end;
        std::vector<DistNode>* dist_nodes;
        const QueryInfo* query_info;
    };
    static void* BthreadRun(void* message);

    void PushSearchResultToContext(GeneralSearchContext* context, const std::vector<DistNode>& dist_vec) const;

    int CalcNodeDist(std::vector<CoarseIndex<SmallBlock>::PostingIterator>& ivf_postings,
                     std::vector<uint32_t>& ivf_postings_group_ids,
                     const QueryInfo& query_info, std::vector<DistNode>& dist_nodes,
                     std::vector<uint32_t>& group_doc_nums);

    int CollectIvfPostings(std::vector<CoarseIndex<SmallBlock>::PostingIterator>& ivf_postings,
                           std::vector<uint32_t>& ivf_postings_group_ids,
                           std::vector<uint32_t>& group_doc_nums,
                           const QueryInfo& query_info,
                           std::vector<std::vector<off_t>>& real_slot_indexs);

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
                       uint32_t offset, bool full_sort = false) const;

    int PostProcess(GeneralSearchContext* context, size_t group_num) const;

    int CollectBasicResult(std::vector<MyHeap<DistNode>>& group_heaps, const std::vector<uint32_t>& topks,
                          GeneralSearchContext* context) const;

    int CollectLeftResult(uint32_t total, std::vector<MyHeap<DistNode>>& group_heaps,
                          GeneralSearchContext* context) const;

    void BatchScore(size_t node_start, size_t node_end,
                    std::vector<DistNode>& dist_nodes,
                    const QueryInfo& query_info   ) const;

    size_t GetPostingsNodeCount(const std::vector<CoarseIndex<SmallBlock>::PostingIterator>& ivf_postings,
                                size_t start, size_t end) const;

private:
    GroupIvfIndex::Pointer index_;
    OrigDistScorer::Factory dist_scorer_factory_;
    std::string index_name_;
    bool delmap_before = false;
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_CORE_GROUP_IVF_SEARCHER_H__
