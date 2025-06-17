#pragma once

#include "src/core/framework/index_framework.h"
#include "src/core/algorithm/general_search_context.h"
#include "src/core/algorithm/searcher.h"
#include "src/core/algorithm/pq_common.h"
#include "disk_vamana_index.h"
#include "src/core/utils/my_heap.h"
#include "src/core/utils/monitor_component.h"

MERCURY_NAMESPACE_BEGIN(core);

class DiskVamanaSearcher : public Searcher
{
public:
    typedef std::shared_ptr<DiskVamanaSearcher> Pointer;

public:
    DiskVamanaSearcher();
    ~DiskVamanaSearcher();

    int Init(IndexParams &params) override;
    int LoadDiskIndex(const std::string& disk_index_path, const std::string& medoids_data_path) override;
    int LoadIndex(const std::string& path) override;
    int LoadIndex(const void* data, size_t size) override;
    void SetIndex(Index::Pointer index) override;
    void SetBaseDocId(exdocid_t baseDocId) override;
    void SetVectorRetriever(const AttrRetriever& retriever) override;
    IndexMeta::FeatureTypes getFType() override;

    int Search(const QueryInfo& query_info, GeneralSearchContext* context = nullptr) override;

    // for monitor
    typedef std::function<void(int /* value */)> metric;
    typedef std::function<void(int /* latency_us */, bool /* succ */)> transaction;

    // 监控平均IO读次数
    metric metric_DiskVamana_MeanIONum;

    // 监控平均缓存命中次数
    metric metric_DiskVamana_MeanCacheHitNum;

    // 监控各操作延时
    transaction transaction_IoRead, transaction_CpuDistCmp;

    Index::Pointer GetIndex() {
        return index_;
    }

private:

    int PostProcess(GeneralSearchContext* context, size_t group_num) const;

private:
    IndexParams params_;
    DiskVamanaIndex::Pointer index_;
};

MERCURY_NAMESPACE_END(core);
