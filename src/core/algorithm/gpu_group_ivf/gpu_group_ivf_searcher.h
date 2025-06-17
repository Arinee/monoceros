#ifdef ENABLE_GPU_IN_MERCURY_
#ifndef __MERCURY_CORE_GPU_GROUP_IVF_SEARCHER_H__
#define __MERCURY_CORE_GPU_GROUP_IVF_SEARCHER_H__

#include "src/core/algorithm/general_search_context.h"
#include "src/core/algorithm/gpu_group_ivf/gpu_group_ivf_batch_task.h"
#include "src/core/algorithm/gpu_group_ivf/gpu_neutron_interface.h"
#include "src/core/algorithm/gpu_group_ivf/gpu_resources_wrapper.h"
#include "src/core/algorithm/group_ivf/group_ivf_index.h"
#include "src/core/algorithm/orig_dist_scorer.h"
#include "src/core/algorithm/query_info.h"
#include "src/core/algorithm/searcher.h"
#include "src/core/framework/index_framework.h"
#include "src/core/framework/utility/thread_pool.h"
#include "src/core/utils/batching_util/basic_batch_scheduler.h"
#include "src/core/utils/monitor_component.h"
#include "src/core/utils/my_heap.h"
#include <memory>
#include <string>
#include <vector>

MERCURY_NAMESPACE_BEGIN(core);

class GpuGroupIvfSearcher : public Searcher
{
public:
    typedef std::shared_ptr<GpuGroupIvfSearcher> Pointer;

public:
    GpuGroupIvfSearcher();
    ~GpuGroupIvfSearcher();
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

    // 监控各类指标
    metric metric_GpuGroupIvf_CentroidNum, metric_GpuGroupIvf_GroupNum, metric_GpuGroupIvf_FullDocNum,
        metric_GpuGroupIvf_RtDocNum, metric_GpuGroupIvf_FullResultDocNum, metric_GpuGroupIvf_RtResultDocNum,
        metric_GpuGroupIvf_BatchConcurrencyNum;
    // 监控各操作延时
    transaction transaction_GpuBatchSize, transaction_GpuProcessTime, transaction_GpuBatchPreProcessTime,
        transaction_GpuBatchProcessTime, transaction_GpuBatchProcessNotifyTime, transaction_GpuPreProcessTime,
        transaction_GpuPostProcessTime, transaction_GpuCollectIvfNextTime, transaction_GpuBatchFirstTaskWaitTime,
        transaction_GpuBatchLastTaskWaitTime, transaction_GpuBatchAllTaskWaitTime, transaction_GpuBatchDataSize;

public:
    int SearchIvf(std::vector<CoarseIndex<SmallBlock>::PostingIterator> &postings, const void *data, size_t size);

    bool Add(docid_t doc_id, const void* data) override;

private:
    void PushSearchResultToContext(GeneralSearchContext *context, std::vector<uint32_t> &dist_nodes,
                                   std::vector<float> &distances, std::vector<int> &labels) const;

    int CollectIvfPostings(const QueryInfo &query_info, std::vector<uint32_t> &group_doc_nums,
                           std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings,
                           std::vector<std::vector<off_t>> &real_slot_indexs,
                           std::vector<uint32_t> &ivf_postings_group_ids, GeneralSearchContext *context);

    int CollectIvfDistNodes(const QueryInfo &query_info, std::vector<uint32_t> &dist_nodes,
                            std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings,
                            GeneralSearchContext *context);

    size_t GetPostingsNodeCount(const std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings, size_t start,
                                size_t end) const;

    int PostProcess(GeneralSearchContext *context, size_t group_num) const;

    // for multi age
    int FillCustomData(char *cpu_data, neutron::gpu::GpuDataParam &gpu_data_param,
                       neutron::gpu::GpuDataOffset &gpu_data_offset,
                       std::vector<neutron::gpu::QueryDataParam> &query_data_params,
                       std::vector<std::vector<uint32_t>> &sort_list_doc_nums, 
                       std::vector<CoarseIndex<SmallBlock>::PostingIterator> &ivf_postings,
                       const QueryInfo &query_info,
                       GeneralSearchContext *context);

private:
    GroupIvfIndex::Pointer index_;
    OrigDistScorer::Factory dist_scorer_factory_;
    neutron::gpu::NeutronIndexInterface *gpu_index_;
    neutron::gpu::NeutronManagerInterface *neutron_manager_interface_;
    bool own_gpu_index_ = false;

private:
    bool enable_batch_ = false;
    bool is_gpu_rt_ = false;
    uint32_t max_build_num_ = 3000000;
    int sort_mode_ = 0;
    std::string index_name_ = "";
    std::string gpu_record_key_ = "";

    std::vector<std::shared_ptr<BasicBatchScheduler<GpuGroupIvfBatchTask>>> basic_batch_schedulers_;

    // for random scheduler
    std::mt19937 random_;
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_CORE_GPU_GROUP_IVF_SEARCHER_H__
#endif // ENABLE_GPU_IN_MERCURY_
