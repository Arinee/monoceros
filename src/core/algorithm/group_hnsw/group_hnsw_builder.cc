#include "group_hnsw_builder.h"
#include "src/core/common/common_define.h"
#include "src/core/algorithm/query_info.h"

MERCURY_NAMESPACE_BEGIN(core);

GroupHnswBuilder::GroupHnswBuilder()
: built_cnt_(0)
{}



int GroupHnswBuilder::Init(IndexParams &params) {
    if (!index_) {
        index_.reset(new GroupHnswIndex());
    }
    if (index_->Create(params) != 0) {
        LOG_ERROR("Failed to init ivf index.");
        return -1;
    }
    sort_build_group_level_ = params.getUint32(PARAM_SORT_BUILD_GROUP_LEVEL);
    return 0;
}

//! Build the index
int GroupHnswBuilder::AddDoc(docid_t doc_id, uint64_t pk,
                             const std::string& build_str, 
                             const std::string& primary_key) {

    if(doc_id == 0) {
        if (index_->InitMappingSpace() != 0) {
            return -1;
        }
    }

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

    const std::vector<GroupInfo>& group_infos = query_info.GetGroupInfos();
    if (group_infos.size() <= 0) {
        LOG_ERROR("query at least has one group");
        return -1;
    }

    // 预处理
    for (auto& group_info : group_infos) {
        if (group_meta_.count(group_info) > 0) {
            group_meta_[group_info]++;
        } else {
            group_meta_[group_info] = 1;
        }
    }

    // 写入doc_id与pk映射关系及向量
    if (index_->BaseIndexAdd(doc_id, pk, query_info.GetVector(), query_info.GetVectorLen())) {
        return -1;
    }

    doc_infos_.emplace_back(std::make_pair(doc_id, group_infos));

    return 0;
}

//! Dump index into file or memory
int GroupHnswBuilder::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) {
    //no use
    return 0;
}

const void * GroupHnswBuilder::DumpIndex(size_t* size) {
    LOG_INFO("GroupHnswBuilder::DumpIndex be called");
    if (index_->CalCoarseIndexCapacity(group_meta_) != 0) {
        return nullptr;
    }
    std::vector<docid_t> doc_ids;
    std::vector<uint32_t> doc_max_layers;
    std::vector<uint32_t> group_doc_ids;
    std::vector<uint64_t> group_offsets;
    for (auto& doc_info : doc_infos_) {

        uint32_t doc_max_layer;
        doc_max_layer = index_->GetRandomLevel();

        // 非HNSW图直接写入group所在内存, HNSW图元信息写入group所在内存
        if (index_->AssignSpace(doc_info.first, doc_info.second, doc_ids, doc_max_layers, group_doc_ids, group_offsets, doc_max_layer) != 0) {
            return nullptr;
        }
        built_cnt_++;
        if ((built_cnt_ % 100000) == 0) {
            LOG_INFO("Current assigned space number: %lu", built_cnt_);
        }
    }

    // 在每个group内build
    for (size_t i = 0; i < doc_ids.size(); i++) {
        mercury::Closure::Pointer task = mercury::Closure::New( this, 
                                                            &GroupHnswBuilder::DoAdd,
                                                            doc_ids.at(i),
                                                            group_doc_ids.at(i),
                                                            group_offsets.at(i),
                                                            index_->GetDocFeature(doc_ids.at(i)),
                                                            doc_max_layers.at(i));
        pool_.enqueue(task, true);
        // DoAdd(group_doc_ids.at(i), group_offsets.at(i), index_->GetDocFeature(doc_info.first), doc_max_layer);
    }

    pool_.waitFinish();

    // 冗余内存裁剪
    index_->RedundantMemClip();

    const void *data = nullptr;
    if (index_->Dump(data, *size) != 0) {
        LOG_ERROR("Failed to dump index.");
        *size = 0;
        return nullptr;
    }
    index_->FreeMem();

    return data;
}

void GroupHnswBuilder::DoAdd(docid_t global_doc_id, docid_t group_doc_id, uint64_t group_offset, const void *val, uint32_t doc_max_layer) {
    if (index_->AddDoc(group_doc_id, group_offset, val, doc_max_layer) != 0) {
        LOG_ERROR("Failed to add into coarse index. doc_id: %u, group_doc_id: %u, group_offset: %lu", global_doc_id, group_doc_id, group_offset);
    }
}

int GroupHnswBuilder::GetRankScore(const std::string& build_str, float * score) {
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

    float query_score = 0.0;
    const std::vector<GroupInfo>& group_infos = query_info.GetGroupInfos();
    for (size_t i = 0; i < group_infos.size(); i++) {
        const GroupInfo& group_info = group_infos.at(i);
        query_score += CalcScore(group_info.level, group_info.id);
    }

    *score = query_score;
    return 0;
}

float GroupHnswBuilder::CalcScore(level_t level, group_t id) {
    return level == sort_build_group_level_ ? log(id + 1) : 0;
}

MERCURY_NAMESPACE_END(core);
