#include "ivf_pq_index.h"
#include "query_distance_matrix.h"
#include "src/core/utils/index_meta_helper.h"
#include "../query_info.h"

MERCURY_NAMESPACE_BEGIN(core);

int16_t GetOrDefault(const IndexParams& params, const std::string& key, const uint16_t default_value) {
    if (params.has(key)) {
        return params.getUint16(key);
    }

    return default_value;
}

int IvfPqIndex::Load(const void* data, size_t size) {
    if (IvfIndex::Load(data, size) != 0) {
        LOG_ERROR("call IvfIndex::Load failed.");
        return -1;
    }

    auto *component = index_package_.get(COMPONENT_INTEGRATE_MATRIX);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_INTEGRATE_MATRIX);
        return -1;
    }

    if (!GetCentroidResource().initIntegrate((void *)component->getData(), component->getDataSize())) {
        LOG_ERROR("centroid resource integrate init error");
        return -1;
    }

    component = index_package_.get(COMPONENT_PQ_CODE_PROFILE);
    if (!component) {
        LOG_ERROR("get component error : %s", COMPONENT_PQ_CODE_PROFILE);
        return -1;
    }

    if (!pq_code_profile_.load((void*)component->getData(), component->getDataSize())) {
        LOG_ERROR("pq code profle load error");
        return -1;
    }

    return 0;
}

int IvfPqIndex::Add(docid_t doc_id, pk_t pk,
                    const std::string& query_str, 
                    const std::string& primary_key) {
    QueryInfo query_info(query_str);
    if (!query_info.MakeAsBuilder()) {
        LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
        return -1;
    }

    return Add(doc_id, pk, query_info.GetVector(), query_info.GetVectorLen());
}

/// add a new vectoV
int IvfPqIndex::Add(docid_t doc_id, pk_t pk, const void *data, size_t size) {
    if (IvfIndex::Add(doc_id, pk, data, size) != 0) {
        LOG_ERROR("call IvfIndex::Add failed. docid: %u, pk: %lu", doc_id, pk);
        return -1;
    }

    QueryDistanceMatrix qdm(index_meta_, &GetCentroidResource());
    std::vector<size_t> level_scan_limit;
    for (size_t i = 0; i < GetCentroidResource().getRoughMeta().levelCnt - 1; ++i) {
        level_scan_limit.push_back(GetCentroidResource().getRoughMeta().centroidNums[i] / 10);
    }
    bool bres = qdm.init(data, level_scan_limit, true);
    if (!bres) {
        LOG_ERROR("Calcualte QDM failed!");
        return -1;
    }
    std::vector<uint16_t> product_labels;
    if (!qdm.getQueryCodeFeature(product_labels)) {
        LOG_ERROR("get query codefeature failed!");
        return -1;
    }

    if (!pq_code_profile_.insert(doc_id, product_labels.data())) {
        LOG_ERROR("Failed to add into pq code profile.");
        return -1;
    }

    return 0;
}

size_t IvfPqIndex::MemQuota2DocCount(size_t mem_quota, size_t elem_size) const {
    int64_t slot_num = GetCentroidResource().getLeafCentroidNum();
    auto fragment_num = GetCentroidResource().getIntegrateMeta().fragmentNum;
    auto product_size = sizeof(uint16_t) * fragment_num; //why uint16_6
    size_t elem_count = 0;
    size_t real_mem_used = 0;
    do {
        elem_count += MIN_BUILD_COUNT;
        real_mem_used = 0;
        real_mem_used += CoarseIndex<BigBlock>::calcSize(slot_num, elem_count);
        real_mem_used += ArrayProfile::CalcSize(elem_count, sizeof(SlotIndex));
        real_mem_used += ArrayProfile::CalcSize(elem_count, product_size);
    } while (real_mem_used <= mem_quota);
    elem_count -= MIN_BUILD_COUNT;
    if (elem_count == 0) {
        LOG_WARN("memQuota: %lu, abnormal elem_count: 0", mem_quota);
    }

    return elem_count;
}

int IvfPqIndex::Create(IndexParams& index_params) {
    Index::SetIndexParams(index_params);
    if (!IndexMetaHelper::parseFrom(index_params.getString(PARAM_DATA_TYPE),
                                    index_params.getString(PARAM_METHOD),
                                    index_params.getUint64(PARAM_DIMENSION),
                                    index_meta_)) {
        LOG_ERROR("Failed to init ivf index meta.");
        return -1;
    }

    if (index_params.getString(PARAM_TRAIN_DATA_PATH) != "") {
        if (!InitPqCentroidMatrix(index_params)) {
            LOG_ERROR("Failed to init pq centroid matrix");
            return -1;
        }
    }

    LOG_INFO("Begin to call IvfIndex::Create.");
    if (IvfIndex::Create(index_params) != 0) {
        LOG_ERROR("call Ivf::Create failed.");
        return -1;
    }
    LOG_INFO("End call IvfIndex::Create.");

    auto fragment_num = GetCentroidResource().getIntegrateMeta().fragmentNum;
    auto product_size = sizeof(uint16_t) * fragment_num; //why uint16_t
    auto product_capacity = ArrayProfile::CalcSize(GetMaxDocNum(), product_size);
    pq_code_base_.assign(product_capacity, 0);
    LOG_INFO("Begin to call pq code profile create");
    if (!pq_code_profile_.create(pq_code_base_.data(), product_capacity, product_size)) {
        LOG_ERROR("Failed to create PQ code profile");
        return -1;
    }
    LOG_INFO("End call pq code profile create");

    return 0;
}

int IvfPqIndex::CopyInit(const Index* index, size_t size) {
    if (IvfIndex::CopyInit(index, size) != 0) {
        LOG_ERROR("call IvfIndex::CopyInit failed.");
        return -1;
    }

    const IvfPqIndex* copy_index = dynamic_cast<const IvfPqIndex*>(index);
    if (!copy_index) {
        LOG_ERROR("Failed to dynamic cast to IvfPqIndex pointer.");
        return -1;
    }

    auto fragment_num = copy_index->GetCentroidResource().getIntegrateMeta().fragmentNum;
    auto product_size = sizeof(uint16_t) * fragment_num; //why uint16_t
    auto product_capacity = ArrayProfile::CalcSize(GetMaxDocNum(), product_size);
    pq_code_base_.assign(product_capacity, 0);
    if (!pq_code_profile_.create(pq_code_base_.data(), product_capacity, product_size)) {
        LOG_ERROR("Failed to create PQ code profile");
        return -1;
    }

    return 0;
}

int IvfPqIndex::CompactIndex() {
    compact_index_ = std::make_shared<IvfPqIndex>();
    size_t total_doc_num = GetDocNum();
    compact_index_->CopyInit(this, total_doc_num);
    LOG_INFO("Total doc num in index: %lu", total_doc_num);

    for (docid_t docid = 0; docid < total_doc_num; docid++) {
        if (CompactEach(docid, compact_index_.get()) != 0) {
            return -1;
        }

        int ret = compact_index_->GetPqCodeProfile().insert(docid, pq_code_profile_.getInfo(docid));
        if (!ret) {
            LOG_ERROR( "insert profile info error with id[%u]", docid);
            return -1;
        }

    }

    return 0;
}

int IvfPqIndex::Dump(const void*& data, size_t& size) {
    if (CompactIndex() != 0) {
        LOG_ERROR("compact index failed.");
        return -1;
    }

    if (DumpHelper::DumpIvfPq(compact_index_.get(), index_package_) != 0) {
        return -1;
    }

    if (!index_package_.dump(data, size)) {
        LOG_ERROR("Failed to dump package.");
        return -1;
    }

    return 0;
}

int64_t IvfPqIndex::UsedMemoryInCurrent() const {
    return IvfIndex::UsedMemoryInCurrent() + sizeof(*this)
        + pq_code_base_.size();
}

bool IvfPqIndex::InitPqCentroidMatrix(const IndexParams& param) {
    uint16_t pq_centroid_num = GetOrDefault(param, PARAM_PQ_CENTROID_NUM, DefaultPqCentroidNum);
    uint16_t pq_fragment_count = GetOrDefault(param, PARAM_PQ_FRAGMENT_NUM, DefaultPqFragmentCnt);

    //init pq
    CentroidResource::IntegrateMeta integrate_meta(index_meta_.sizeofElement() / pq_fragment_count,
                                                   pq_fragment_count, pq_centroid_num);
    if (!GetCentroidResource().create(integrate_meta)) {
        LOG_ERROR("Failed to create integrate meta.");
        return false;
    }

    size_t pq_element_size = index_meta_.sizeofElement() / pq_fragment_count;
    //pq, 遍历所有分片
    for (int i = 0; i < pq_fragment_count; i++) {
        size_t centroid_size = 0;
        std::string file_path = param.getString(PARAM_TRAIN_DATA_PATH) + PQ_CENTROID_FILE_MIDDLEFIX
            + std::to_string(i) + PQ_CENTROID_FILE_POSTFIX;
        MatrixPointer matrix_pointer = DoLoadCentroidMatrix(file_path,
                                                            index_meta_.dimension() / pq_fragment_count,
                                                            pq_element_size,
                                                            centroid_size);
        if (!matrix_pointer) {
            LOG_ERROR("Failed to Load centroid Matrix.");
            return false;
        }
        char* centroid_matrix = matrix_pointer.get();

        size_t j = 0;
        for (; j < centroid_size; j++) {
            if (!GetCentroidResource().setValueInIntegrateMatrix(i, j, centroid_matrix + j * pq_element_size)) {
                LOG_ERROR("Failed to set centroid resource rough matrix.");
                return false;
            }
        }

        //if not enough pq centroids, use last to make up
        for (; j < pq_centroid_num; j++) {
            if (!GetCentroidResource().setValueInIntegrateMatrix(i, j, centroid_matrix + (centroid_size - 1) * pq_element_size)) {
                LOG_ERROR("Failed to make up centroid resource rough matrix.");
                return false;
            }
        }
    }

    return true;
}

MERCURY_NAMESPACE_END(core);
