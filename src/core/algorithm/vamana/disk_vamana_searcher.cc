#include "disk_vamana_searcher.h"
#include <errno.h>
#include <pthread.h>

MERCURY_NAMESPACE_BEGIN(core);

DiskVamanaSearcher::DiskVamanaSearcher()
{
    if (!index_) {
        index_.reset(new DiskVamanaIndex());
    }
};

DiskVamanaSearcher::~DiskVamanaSearcher(){};

int DiskVamanaSearcher::Init(IndexParams &params)
{
    index_->SetIndexParams(params);
    params_ = params;

    MONITOR_METRIC(DiskVamana_MeanIONum);
    MONITOR_METRIC(DiskVamana_MeanCacheHitNum);

    MONITOR_TRANSACTION(DiskVamana, IoRead);
    MONITOR_TRANSACTION(DiskVamana, CpuDistCmp);

    return 0;
}

int DiskVamanaSearcher::LoadIndex(const std::string &path)
{
    // no use
    return -1;
}

int DiskVamanaSearcher::LoadDiskIndex(const std::string& disk_index_path, const std::string& medoids_data_path)
{
    index_->LoadDiskIndex(disk_index_path, medoids_data_path);

    uint32_t num_nodes_to_cache = (uint64_t)(std::round(index_->GetDocNum() * defaults::CACHE_RATIO));

    num_nodes_to_cache = 0; // TODO: fix cache memleak

    LOG_INFO("Trying to Cache %d nodes around medoid(s)", num_nodes_to_cache);

    std::vector<uint32_t> node_list;

    index_->cache_bfs_levels(num_nodes_to_cache, node_list);

    index_->load_cache_list(node_list);

    node_list.clear();
    
    node_list.shrink_to_fit();

    return 0;
}

int DiskVamanaSearcher::LoadIndex(const void *data, size_t size)
{
    if (index_->Load(data, size) != 0) {
        LOG_ERROR("Load disk_vamana_index failed in searcher");
        return -1;
    }

    return 0;
}

void DiskVamanaSearcher::SetIndex(Index::Pointer index)
{
    index_ = std::dynamic_pointer_cast<DiskVamanaIndex>(index);
}

void DiskVamanaSearcher::SetBaseDocId(exdocid_t baseDocId)
{
    base_docid_ = baseDocId;
}

void DiskVamanaSearcher::SetVectorRetriever(const AttrRetriever &retriever)
{
    vector_retriever_ = retriever;
}

IndexMeta::FeatureTypes DiskVamanaSearcher::getFType()
{
    return this->index_->GetIndexMeta().type();
}

int DiskVamanaSearcher::Search(const QueryInfo &query_info, GeneralSearchContext *context)
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

    uint32_t L = defaults::SEARCH_LIST_SIZE;

    uint32_t BW = defaults::SEARCH_BEAM_WIDTH;

    if (index_->GetIndexParams().has(PARAM_VAMANA_INDEX_SEARCH_L)) {
        L = index_->GetIndexParams().getInt32(PARAM_VAMANA_INDEX_SEARCH_L);
    }

    if (index_->GetIndexParams().has(PARAM_VAMANA_INDEX_SEARCH_BW)) {
        BW = index_->GetIndexParams().getInt32(PARAM_VAMANA_INDEX_SEARCH_BW);
    }

    if (query_info.GetContextParams().has(PARAM_VAMANA_INDEX_SEARCH_L)) {
        L = query_info.GetContextParams().getInt32(PARAM_VAMANA_INDEX_SEARCH_L);
    }

    if (query_info.GetContextParams().has(PARAM_VAMANA_INDEX_SEARCH_BW)) {
        BW = query_info.GetContextParams().getInt32(PARAM_VAMANA_INDEX_SEARCH_BW);
    }

    float downgrade_percent = query_info.GetContextParams().has(PARAM_DOWNGRADE_PERCENT)
                                  ? query_info.GetContextParams().getFloat(PARAM_DOWNGRADE_PERCENT)
                                  : 1;
    L = L * downgrade_percent;

    if (topk > L) {
        topk = L;
    }

    if (query_info.GetDimension() != index_->GetIndexMeta().dimension()) {
        LOG_ERROR("query dimension %lu != index dimension %lu.", query_info.GetDimension(), index_->GetIndexMeta().dimension());
        return -1;
    }
    
    if (getFType() == mercury::core::IndexMeta::kTypeHalfFloat) {
        QueryInfo query_info_raw(query_info.GetRawQuery());
        
        if (!query_info_raw.MakeAsSearcher()) {
            LOG_ERROR("resolve query failed. query str:%s", query_info_raw.GetRawQuery().c_str());
            return -1;
        }

        index_->cached_beam_search_half(query_info.GetVector(), query_info_raw.GetVector(), topk, L,
                                    query_result_ids_64.data(),
                                    query_result_dist.data(),
                                    BW, stats);
    } else {
        index_->cached_beam_search(query_info.GetVector(), topk, L,
                                query_result_ids_64.data(),
                                query_result_dist.data(),
                                BW, stats);
    }

    for(uint32_t i = 0; i < topk; i++) {
        context->emplace_back(0, query_result_ids_64[i], query_result_dist[i]);
    }

    MONITOR_METRIC_LOG(DiskVamana_MeanIONum, stats->n_ios);

    MONITOR_METRIC_LOG(DiskVamana_MeanCacheHitNum, stats->n_cache_hits);

    transaction_IoRead(stats->io_us, true);

    transaction_CpuDistCmp(stats->cpu_us, true);

    PostProcess(context, 0);
    
    return 0;
}

int DiskVamanaSearcher::PostProcess(GeneralSearchContext *context, size_t group_num) const
{
    std::vector<SearchResult> &results = context->Result();
    std::sort(results.begin(), results.end(),
              [](const SearchResult &a, const SearchResult &b) { return a.gloid < b.gloid; });

    return 0;
}

MERCURY_NAMESPACE_END(core);
