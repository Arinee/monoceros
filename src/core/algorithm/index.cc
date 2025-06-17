#include "index.h"
#include "query_info.h"

MERCURY_NAMESPACE_BEGIN(core);

int Index::Load(const void* data, size_t size) {
    if (!index_package_.load(data, size)) {
        LOG_ERROR("Failed to load index pacakge");
        return -1;
    }

    auto *component = index_package_.get(COMPONENT_FEATURE_META);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_FEATURE_META);
        return -1;
    }
    if (!index_meta_.deserialize(component->getData(), component->getDataSize())) {
        LOG_ERROR("index meta deserialize error");
        return -1;
    }

    // Read pk profile
    if (WithPk()) {
        auto *component = index_package_.get(COMPONENT_PK_PROFILE);
        if (!component) {
            LOG_ERROR("get component %s error", COMPONENT_PK_PROFILE);
            return -1;
        }
        if (!pk_profile_.load((void*)component->getData(), component->getDataSize())) {
            LOG_ERROR("pk profile load error");
            return -1;
        }

        // Read IDMap
        component = index_package_.get(COMPONENT_IDMAP);
        if (!component) {
            LOG_ERROR("get component %s error", COMPONENT_IDMAP);
            return -1;
        }
        if (0 != id_map_.mount((char*)component->getData(), component->getDataSize())) {
            LOG_ERROR("idmap load error");
            return -1;
        }
    }

    return 0;
}

int Index::CopyInit(const Index* index, size_t doc_num) {
    if (!index) {
        LOG_ERROR("Invalid index pointer.");
        return -1;
    }

    max_doc_num_ = doc_num;
    index_meta_ = index->index_meta_;
    index_params_ = index->index_params_;

    if (Create(max_doc_num_) != 0) {
        LOG_ERROR("Failed to call index::create.");
        return -1;
    }
    return 0;
}

int Index::Create(size_t max_doc_num) {
    max_doc_num_ = max_doc_num;
    if (WithPk()) {
        size_t capacity = ArrayProfile::CalcSize(max_doc_num_, sizeof(pk_t));
        pk_profile_base_.assign(capacity, 0);
        bool res = pk_profile_.create(pk_profile_base_.data(), capacity, sizeof(pk_t));
        if (!res) {
            LOG_ERROR("create pk_profile failed. max_doc_num: %lu, capacity: %lu", max_doc_num, capacity);
            return -1;
        }

        capacity = HashTable<pk_t, docid_t>::needMemSize(max_doc_num);
        id_map_base_.assign(capacity, 0);
        int ret = id_map_.mount(id_map_base_.data(), capacity, max_doc_num_, true);
        if (ret != 0) {
            LOG_ERROR("create id_map failed. max_doc_num: %lu, capacity: %lu", max_doc_num, capacity);
            return -1;
        }
    }

    return 0;
}

int Index::Add(docid_t doc_id, pk_t pk, const void *val, size_t len) {
    if (WithPk()) {
        if (!pk_profile_.insert(doc_id, &pk)) {
            LOG_ERROR("Failed to add into pk profile. doc_id:%u, pk: %lu", doc_id, pk);
            return -1;
        }

        if (id_map_.insert(pk, doc_id) != 0) {
            LOG_ERROR("Failed to add into id_map. doc_id:%u, pk: %lu", doc_id, pk);
            return -1;
        }
    }

    return 0;
}

int Index::Add(docid_t doc_id, pk_t pk,
               const std::string& query_str, 
               const std::string& primary_key) {
    QueryInfo query_info(query_str);
    if (index_meta_.type() == IndexMeta::FeatureTypes::kTypeBinary) {
        query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeBinary);
    }

    if (index_meta_.type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
        query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
    }

    if (!query_info.MakeAsBuilder()) {
        LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
        return -1;
    }

    return Add(doc_id, pk, query_info.GetVector(), query_info.GetVectorLen());
}

MERCURY_NAMESPACE_END(core);
