#include "ram_vamana_builder.h"
#include "src/core/common/common_define.h"

MERCURY_NAMESPACE_BEGIN(core);

RamVamanaBuilder::RamVamanaBuilder() {}

int RamVamanaBuilder::Init(IndexParams &params) {
    if (!index_) {
        index_.reset(new RamVamanaIndex());
    }

    if (index_->Create(params) != 0) {
        LOG_ERROR("Failed to init ram vamana index.");
        return -1;
    }

    return 0;
}

//! Build the index
int RamVamanaBuilder::AddDoc(docid_t doc_id, uint64_t pk,
                             const std::string& build_str, 
                             const std::string& primary_key) {

    QueryInfo query_info(build_str);

    if (index_->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
        query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
    }

    if (!query_info.MakeAsBuilder()) {
        LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
        return -1;
    }

    if (index_->BaseIndexAdd(doc_id, query_info.GetVector())) {
        return -1;
    }

    return 0;
}

//! Dump index into file or memory
int RamVamanaBuilder::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) {
    //no use
    return 0;
}

const void * RamVamanaBuilder::DumpIndex(size_t* size) {

    // dump index to a data pointer and return 
    const void *data = nullptr;
    if (index_->Dump(data, *size) != 0) {
        LOG_ERROR("Failed to dump index.");
        *size = 0;
        return nullptr;
    }

    return data;
}

int RamVamanaBuilder::DumpRamVamanaIndex(std::string &path_prefix) {
    index_->BuildMemIndex();
    index_->DumpMemLocal(path_prefix);
    if (file_exists(path_prefix) && file_exists(path_prefix + ".data")) {
        return 0;
    }
    return -1;
}


int RamVamanaBuilder::GetRankScore(const std::string& build_str, float * score) {
    return 0;
}

MERCURY_NAMESPACE_END(core);
