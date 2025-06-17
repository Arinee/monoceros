#ifndef __MERCURY_CORE_GROUP_HNSW_SEARCHER_H__
#define __MERCURY_CORE_GROUP_HNSW_SEARCHER_H__

#include "src/core/framework/index_framework.h"
#include "src/core/algorithm/general_search_context.h"
#include "src/core/algorithm/searcher.h"
#include "src/core/algorithm/pq_common.h"
#include "group_hnsw_index.h"
#include "src/core/utils/my_heap.h"
#include "src/core/utils/monitor_component.h"

MERCURY_NAMESPACE_BEGIN(core);

class GroupHnswSearcher : public Searcher
{
public:
    typedef std::shared_ptr<GroupHnswSearcher> Pointer;

public:
    GroupHnswSearcher();
    ~GroupHnswSearcher();

    int Init(IndexParams &params) override;
    int LoadIndex(const std::string& path) override;
    int LoadIndex(const void* data, size_t size) override;
    void SetIndex(Index::Pointer index) override;
    void SetBaseDocId(exdocid_t baseDocId) override;
    void SetVectorRetriever(const AttrRetriever& retriever) override;
    IndexMeta::FeatureTypes getFType() override;
    //! search by query
    // 类目层级:c1#topk,c2#topk;类目层级:c3#topk||v1 v2 v3...
    // 1:111#500,112#500;2:222#100||0.1 0.3 0.2
    int Search(const QueryInfo& query_info, GeneralSearchContext* context = nullptr) override;

    typedef std::function<void(int /* value */)> metric;
    metric metric_GroupHnsw_GroupNum, metric_GroupHnsw_BruteCmpCnt, metric_GroupHnsw_HnswCmpCnt;

private:

    struct SearcherMessage {
        GroupHnswIndex* index;
        const GroupInfo* group_info;
        uint32_t topk;
        const void * query_vector;
        size_t vector_length;
        GeneralSearchContext* context;
        MyHeap<DistNode>* group_heap;
        int max_scan_num_in_query;
        std::pair<int, int>* cmp_cnt;
    };

    int CollectGroupResults(const QueryInfo& query_info,
                            std::vector<MyHeap<DistNode>>& group_heaps,
                            GeneralSearchContext* context) const;
    int CollectBasicResult(std::vector<MyHeap<DistNode>>& group_heaps,
                           const std::vector<uint32_t>& topks,
                           GeneralSearchContext* context) const;
    int CollectLeftResult(uint32_t total,
                          std::vector<MyHeap<DistNode>>& group_heaps,
                          GeneralSearchContext* context) const;

    int PostProcess(GeneralSearchContext* context, size_t group_num) const;

    static void* BthreadRun(void* message);

private:
    IndexParams params_;
    GroupHnswIndex::Pointer index_;
    uint64_t part_dimension_;
};

MERCURY_NAMESPACE_END(core);

#endif //__MERCURY_CORE_GROUP_HNSW_SEARCHER_H__
