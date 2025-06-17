#include "ram_vamana_index.h"
#include "src/core/utils/index_meta_helper.h"
#include "src/core/common/common.h"

MERCURY_NAMESPACE_BEGIN(core);

RamVamanaIndex::RamVamanaIndex()
    : R_(64),
      L_(100),
      T_(1)
{

}

RamVamanaIndex::~RamVamanaIndex() {}

int RamVamanaIndex::Create(IndexParams& index_params) {
    
    Index::SetIndexParams(index_params);

    if (!IndexMetaHelper::parseFrom(index_params.getString(PARAM_DATA_TYPE),
                                    index_params.getString(PARAM_METHOD),
                                    index_params.getUint64(PARAM_DIMENSION),
                                    index_meta_)) {
        LOG_ERROR("Failed to init DiskVamana index meta.");
        return -1;
    }

    // determine data type
    if (index_meta_.type() == mercury::core::IndexMeta::kTypeHalfFloat) {
        _use_half = true;
    }

    // set Vamana graph index max degree (R)
    if (index_params.has(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE)) {
        R_ = index_params.getUint32(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE);
        if (R_ <= 0) {
            LOG_ERROR("mercury.vamana.index.max_graph_degree must larger than 0");
            return -1;
        }
    }

    // set Vamana graph index build max size of search list (L)
    if (index_params.has(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST)) {
        L_ = index_params.getUint32(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST);
        if (L_ <= 0) {
            LOG_ERROR("mercury.vamana.index.build.max_search_list must larger than 0");
            return -1;
        }
    }

    if (index_params.has(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM)) {
        T_ = index_params.getUint32(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM);
        if (T_ <= 0) {
            LOG_ERROR("mercury.vamana.index.build.thread_num must larger than 0");
            return -1;
        }
    }

    //设置doc数量
    size_t max_build_num = index_params.getUint64(PARAM_GENERAL_MAX_BUILD_NUM);
    if (max_build_num > 0) {
        max_doc_num_ = max_build_num;
    } else {
        LOG_ERROR("Not set param: mercury.general.index.max_build_num");
        return -1;
    }

    data_dim_ = index_meta_.dimension();

    data_size_ = (uint16_t)index_meta_.sizeofElement() / data_dim_;

    size_t max_reserve_degree = (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 * R_);

    auto index_build_params = IndexWriteParametersBuilder(L_, R_)
                                .with_alpha(defaults::ALPHA)
                                .with_saturate_graph(true)
                                .with_num_threads(T_)
                                .build();

    auto config = IndexConfigBuilder()
                    .with_dimension(data_dim_)
                    .with_max_points(max_doc_num_)
                    .with_max_reserve_degree(max_reserve_degree)
                    .is_dynamic_index(false)
                    .with_index_write_params(index_build_params)
                    .is_enable_tags(false)
                    .build();

    coarseVamanaIndex_ = std::make_unique<CoarseVamanaIndex>(data_size_, config, index_meta_.method(), _use_half);

    return 0;
}

int RamVamanaIndex::BaseIndexAdd(docid_t doc_id, const void *val)
{
    coarseVamanaIndex_->set_data_vec(doc_id, val);

    return 0;
}

void RamVamanaIndex::BuildMemIndex()
{
    coarseVamanaIndex_->build_with_data_populated();
}

void RamVamanaIndex::DumpMemLocal(const std::string& ram_index_path)
{
    coarseVamanaIndex_->save(ram_index_path.c_str());
}

int RamVamanaIndex::Dump(const void*& data, size_t& size)
{
    // dump common
    if (DumpHelper::DumpCommon(this, index_package_) != 0) {
        LOG_ERROR("dump into package failed.");
        return -1;
    }

    index_package_.emplace(COMPONENT_VAMANA_NEIGHBOR_R, &R_, sizeof(R_));

    index_package_.emplace(COMPONENT_VAMANA_CADIDATE_L, &L_, sizeof(L_));

    if (!index_package_.dump(data, size)) {
        LOG_ERROR("Failed to dump package.");
        return -1;
    }

    LOG_INFO("RamVamanaIndex::Dump End! index_package_ size: %lu", size);

    return 0;
}

int RamVamanaIndex::Load(const void* data, size_t size) {
    LOG_INFO("Begin to Load ram_vamana_index. size: %lu.", size);
    if (Index::Load(data, size) != 0) {
        LOG_ERROR("Failed to call Index::Load.");
        return -1;
    }

    data_dim_ = index_meta_.dimension();

    data_size_ = index_meta_.sizeofElement() / index_meta_.dimension();

    // determine data type
    if (index_meta_.type() == mercury::core::IndexMeta::kTypeHalfFloat) {
        _use_half = true;
    }

    if (coarseVamanaIndex_ == nullptr) {

        auto *component = index_package_.get(COMPONENT_VAMANA_NEIGHBOR_R);
        if (!component) {
            LOG_ERROR("get component error: %s", COMPONENT_VAMANA_NEIGHBOR_R);
            return -1;
        }
        R_ = *((uint32_t*)component->getData());

        component = index_package_.get(COMPONENT_VAMANA_CADIDATE_L);
        if (!component) {
            LOG_ERROR("get component error: %s", COMPONENT_VAMANA_CADIDATE_L);
            return -1;
        }
        L_ = *((uint32_t*)component->getData());


        size_t max_reserve_degree = (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 * R_);

        auto index_searcher_params = IndexWriteParametersBuilder(L_, R_)
                                    .with_alpha(defaults::ALPHA)
                                    .build();

        auto config = IndexConfigBuilder()
                        .with_dimension(data_dim_)
                        .with_max_reserve_degree(max_reserve_degree)
                        .is_dynamic_index(false)
                        .with_index_write_params(index_searcher_params)
                        .is_enable_tags(false)
                        .build();
        
        coarseVamanaIndex_ = std::make_unique<CoarseVamanaIndex>(data_size_, config, index_meta_.method(), _use_half);
    }

    return 0;
}

void RamVamanaIndex::LoadMemLocal(const std::string& ram_index_path) {
    uint32_t num_threads = defaults::MAX_RAM_SEARCH_THREAD_NUM;
    uint32_t search_l = defaults::MAX_RAM_SEARCH_LIST_SIZE;
    coarseVamanaIndex_->load(ram_index_path.c_str(), num_threads, search_l);
}

void RamVamanaIndex::Search(const void *query, size_t K, uint32_t L, uint64_t *indices,
                            float *distances, uint32_t &num_cmps) 
{
    num_cmps = coarseVamanaIndex_->search(query, K, L, indices, distances).second;
}

void RamVamanaIndex::GetBaseVec(docid_t doc_id, void *dest)
{
    coarseVamanaIndex_->get_data_vec(doc_id, dest);
}

size_t RamVamanaIndex::GetBaseVecNum()
{
    return coarseVamanaIndex_->get_active_num();
}

uint32_t RamVamanaIndex::GetL()
{
    return L_;
}

MERCURY_NAMESPACE_END(core);