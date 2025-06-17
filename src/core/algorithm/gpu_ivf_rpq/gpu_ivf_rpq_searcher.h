#ifdef ENABLE_GPU_IN_MERCURY_
#ifndef __MERCURY_CORE_GPU_IVF_RPQ_SEARCHER_H__
#define __MERCURY_CORE_GPU_IVF_RPQ_SEARCHER_H__

#include "src/core/algorithm/general_search_context.h"
#include "src/core/algorithm/gpu_group_ivf/gpu_neutron_interface.h"
#include "src/core/algorithm/gpu_group_ivf/gpu_resources_wrapper.h"
#include "src/core/algorithm/ivf_rpq/ivf_rpq_index.h"
#include "src/core/algorithm/ivf_rpq/pq_dist_scorer1.h"
#include "src/core/algorithm/orig_dist_scorer.h"
#include "src/core/algorithm/partition_strategy.h"
#include "src/core/algorithm/searcher.h"
#include "src/core/framework/index_framework.h"
#include "src/core/framework/utility/thread_pool.h"
#include "src/core/utils/monitor_component.h"
#include "src/core/utils/my_heap.h"
#include <memory>
#include <string>
#include <vector>

MERCURY_NAMESPACE_BEGIN(core);

void PushToHeap(MyHeap<DistNode> &dist_heap, MyHeap<DistNode> &left_result);

class GpuIvfRpqSearcher : public Searcher
{
public:
    typedef std::shared_ptr<GpuIvfRpqSearcher> Pointer;

public:
    GpuIvfRpqSearcher();
    ~GpuIvfRpqSearcher();
    //! Init from params
    int Init(IndexParams &params) override;

    int LoadIndex(const std::string &path) override;
    int LoadIndex(const void *data, size_t size) override;
    void SetIndex(Index::Pointer index) override;
    void SetBaseDocId(exdocid_t baseDocId) override;
    IndexMeta::FeatureTypes getFType() override;
    //! search by query
    int Search(const QueryInfo &query_info, GeneralSearchContext *context = nullptr);

    typedef std::function<void(int /* latency_us */, bool /* succ */)> transaction;
    typedef std::function<void(int /* value */)> metric;

    // 监控各操作延时
    transaction transaction_GpuProcessTime, transaction_GpuPreProcessTime, transaction_GpuPostProcessTime;
    // 监控访问中心点数centroid_num，类目数group_num，文档数doc_num
    metric metric_GpuIvfRpq_CentroidNum, metric_GpuIvfRpq_GroupNum, metric_GpuIvfRpq_FullDocNum,
        metric_GpuIvfRpq_RtDocNum;

public:
    int SearchIvf(std::vector<CoarseIndex<SmallBlock>::PostingIterator> &postings, const void *data, size_t size);

private:
    static void *BthreadRun(void *message);

    void PushSearchResultToContext(GeneralSearchContext *context, const std::vector<DistNode> &dist_vec) const;

    int CollectIvfPostings(std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings,
                           std::vector<uint32_t> &ivf_postings_group_ids, std::vector<uint32_t> &group_doc_nums,
                           const QueryInfo &query_info, std::vector<std::vector<off_t>> &real_slot_indexs,
                           std::vector<std::pair<uint32_t, distance_t>>& lvl1_idx_dis);
                           
    void RecoverPostingFromSlot(std::vector<CoarseIndex<SmallBlock>::PostingIterator> &postings,
                                std::vector<uint32_t> &ivf_postings_group_ids,
                                std::vector<std::vector<off_t>> &real_slot_indexs,
                                std::vector<uint32_t> &group_doc_nums, bool need_truncate, bool is_multi_query);

    int CollectIvfDistNodes(const QueryInfo &query_info, std::vector<uint32_t> &dist_nodes,
                            std::vector<uint32_t> &lvl1_indices, std::vector<distance_t> &lvl1_dists,
                            std::vector<std::pair<uint32_t, distance_t>>& lvl1_idx_dis,
                            std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings,
                            uint32_t total_node_count, GeneralSearchContext *context);

    int PostProcess(const QueryInfo &query_info, GeneralSearchContext *context, size_t group_num) const;

private:
    IvfRpqIndex::Pointer index_;
    OrigDistScorer::Factory dist_scorer_factory_;
    neutron::gpu::NeutronIndexInterface *gpu_index_;
    neutron::gpu::NeutronManagerInterface *neutron_manager_interface_;
    bool own_gpu_index_ = false;
    int sort_mode_ = 0;
    std::string index_name_ = "";
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_CORE_GPU_IVF_RPQ_SEARCHER_H__
#endif // ENABLE_GPU_IN_MERCURY_