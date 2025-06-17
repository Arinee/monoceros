#include "ivf_builder.h"
#include "../query_info.h"

MERCURY_NAMESPACE_BEGIN(core);

//! Initialize Builder
int IvfBuilder::Init(IndexParams &params) {
    if (!index_) {
        index_.reset(new IvfIndex());
    }
    if (index_->Create(params) != 0) {
        LOG_ERROR("Failed to init ivf index.");
        return -1;
    }

    return 0;
}

//! Build the index
int IvfBuilder::AddDoc(docid_t doc_id, uint64_t pk,
                       const std::string& build_str, 
                       const std::string& primary_key) {
    return index_->Add(doc_id, pk, build_str);
}

int IvfBuilder::GetRankScore(const std::string& build_str, float * score) {
    QueryInfo query_info(build_str);
    if (index_->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeBinary) {
        query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeBinary);
    }

    if (index_->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
        query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
    }

    if (!query_info.MakeAsBuilder()) {
        LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
        return -1;
    }

    SlotIndex slot_index = index_->GetNearestLabel(query_info.GetVector(), query_info.GetVectorLen());
    if (slot_index == INVALID_SLOT_INDEX) {
        return -1;
    }
    *score = (float)(slot_index);
    return 0;
}

//! Dump index into file or memory
int IvfBuilder::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) {
    //TODO
    return 0;
}

const void * IvfBuilder::DumpIndex(size_t* size) {
    const void *data = nullptr;
    if (index_->Dump(data, *size) != 0) {
        LOG_ERROR("Failed to dump index.");
        *size = 0;
        return nullptr;
    }

    return data;
}

MERCURY_NAMESPACE_END(core);
