#pragma once

#include "src/core/framework/index_framework.h"
#include "src/core/algorithm/general_search_context.h"
#include "src/core/algorithm/searcher.h"
#include "src/core/algorithm/pq_common.h"
#include "ram_vamana_index.h"
#include "src/core/utils/my_heap.h"
#include "src/core/utils/monitor_component.h"

MERCURY_NAMESPACE_BEGIN(core);

class RamVamanaSearcher : public Searcher
{
public:
    typedef std::shared_ptr<RamVamanaSearcher> Pointer;

public:
    RamVamanaSearcher();
    ~RamVamanaSearcher();

    int Init(IndexParams &params) override;
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

    Index::Pointer GetIndex() {
        return index_;
    }

private:

    int PostProcess(GeneralSearchContext* context, size_t group_num) const;

private:
    IndexParams params_;
    RamVamanaIndex::Pointer index_;
};

MERCURY_NAMESPACE_END(core);
