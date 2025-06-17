#include "ram_vamana_searcher.h"
#include <errno.h>
#include <pthread.h>

MERCURY_NAMESPACE_BEGIN(core);

RamVamanaSearcher::RamVamanaSearcher()
{
    if (!index_) {
        index_.reset(new RamVamanaIndex());
    }
};

RamVamanaSearcher::~RamVamanaSearcher(){};

int RamVamanaSearcher::Init(IndexParams &params)
{
    index_->SetIndexParams(params);
    params_ = params;
    return 0;
}

int RamVamanaSearcher::LoadIndex(const std::string &path)
{
    index_->LoadMemLocal(path);
    return 0;
}

int RamVamanaSearcher::LoadIndex(const void *data, size_t size)
{
    if (index_->Load(data, size) != 0) {
        LOG_ERROR("Load ram_vamana_index failed in searcher");
        return -1;
    }

    return 0;
}

void RamVamanaSearcher::SetIndex(Index::Pointer index)
{
    index_ = std::dynamic_pointer_cast<RamVamanaIndex>(index);
}

void RamVamanaSearcher::SetBaseDocId(exdocid_t baseDocId)
{
    base_docid_ = baseDocId;
}

void RamVamanaSearcher::SetVectorRetriever(const AttrRetriever &retriever)
{
    vector_retriever_ = retriever;
}

IndexMeta::FeatureTypes RamVamanaSearcher::getFType()
{
    return this->index_->GetIndexMeta().type();
}

int RamVamanaSearcher::Search(const QueryInfo &query_info, GeneralSearchContext *context)
{    
    const std::vector<uint32_t> &topks = query_info.GetTopks();
    uint32_t topk = topks[0];
    const uint32_t total = query_info.GetTotalRecall();
    if (topk == 0) {
        topk = total;
    }
    context->Result().reserve(topk);
    std::vector<uint64_t> query_result_ids_64(topk);
    std::vector<float> query_result_dist(topk);
    auto stats = new QueryStats;

    uint32_t L = index_->GetL();

    if (query_info.GetContextParams().has(PARAM_VAMANA_INDEX_SEARCH_L)) {
        L = query_info.GetContextParams().getInt32(PARAM_VAMANA_INDEX_SEARCH_L);
    }
    float downgrade_percent = query_info.GetContextParams().has(PARAM_DOWNGRADE_PERCENT)
                                  ? query_info.GetContextParams().getFloat(PARAM_DOWNGRADE_PERCENT)
                                  : 1;
    L = L * downgrade_percent;

    if (topk > L) {
        topk = L;
    }

    index_->Search(query_info.GetVector(), topk, L, query_result_ids_64.data(), query_result_dist.data(), stats->n_cmps);

    for(uint32_t i = 0; i < topk; i++) {
        context->emplace_back(0, query_result_ids_64[i], query_result_dist[i]);
    }

    PostProcess(context, 0);
    
    return 0;
}

int RamVamanaSearcher::PostProcess(GeneralSearchContext *context, size_t group_num) const
{
    std::vector<SearchResult> &results = context->Result();
    std::sort(results.begin(), results.end(),
              [](const SearchResult &a, const SearchResult &b) { return a.gloid < b.gloid; });

    return 0;
}

MERCURY_NAMESPACE_END(core);
