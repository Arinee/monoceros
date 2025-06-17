#include "group_ivf_pq_small_builder.h"
#include "src/core/algorithm/query_info.h"
#include <math.h>

MERCURY_NAMESPACE_BEGIN(core);

//! Initialize Builder
int GroupIvfPqSmallBuilder::Init(IndexParams &params)
{
    if (!index_) {
        index_.reset(new GroupIvfPqSmallIndex());
    }
    if (index_->Create(params) != 0) {
        LOG_ERROR("Failed to init ivf index.");
        return -1;
    }

    // default value 0 when param not set
    sort_build_group_level_ = params.getUint32(PARAM_SORT_BUILD_GROUP_LEVEL);

    return 0;
}

//! Build the index
int GroupIvfPqSmallBuilder::AddDoc(docid_t doc_id, uint64_t pk, const std::string &build_str, const std::string &primary_key)
{
    return index_->Add(doc_id, pk, build_str, primary_key);
}

int GroupIvfPqSmallBuilder::GetRankScore(const std::string &build_str, float *score)
{
    QueryInfo query_info(build_str);

    if (index_->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
        query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
    }

    if (!query_info.MakeAsBuilder()) {
        LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
        return -1;
    }

    float query_score = 0.0;
    const std::vector<GroupInfo> &group_infos = query_info.GetGroupInfos();

    for (size_t i = 0; i < group_infos.size(); i++) {
        const GroupInfo &group_info = group_infos.at(i);
        gindex_t group_index = index_->GetGroupManager().GetGroupIndex(group_info);
        if (group_index == INVALID_GROUP_INDEX) {
            LOG_ERROR("no such group in index. level: %d, id: %d", group_info.level, group_info.id);
            continue;
        }

        // 一层聚类
        SlotIndex label = index_->GetNearestGroupLabel(query_info.GetVector(), query_info.GetVectorLen(), group_index,
                                                       index_->GetCentroidResourceManager());
        if (label == INVALID_SLOT_INDEX) {
            return -1;
        }
        // 二层聚类
        if (index_->EnableFineCluster()) {
            label = index_->GetNearestGroupLabel(query_info.GetVector(), query_info.GetVectorLen(), label,
                                                 index_->GetFineCentroidResourceManager());
            if (label == INVALID_SLOT_INDEX) {
                return -1;
            }
        }

        query_score += CalcScore(group_info.level, label);
    }

    *score = query_score;
    return 0;
}

// 对于指定层级, 挂在index较小的中心点下面的分数应该大于index较大的中心点
float GroupIvfPqSmallBuilder::CalcScore(uint32_t level, SlotIndex label)
{
    return level == sort_build_group_level_ ? (label + 1) : 0;
}

//! Dump index into file or memory
int GroupIvfPqSmallBuilder::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg)
{
    // TODO
    return 0;
}

const void *GroupIvfPqSmallBuilder::DumpIndex(size_t *size)
{
    const void *data = nullptr;
    if (index_->Dump(data, *size) != 0) {
        LOG_ERROR("Failed to dump index.");
        *size = 0;
        return nullptr;
    }

    return data;
}

MERCURY_NAMESPACE_END(core);
